import sys
sys.path.append('..')
import os
import glob
#import horovd.torch as hvd
import torch
import torchbnn as bnn
import pandas as pd
import numpy as np
import pickle
from sklearn.covariance import MinCovDet
import torch.nn as nn
from torch.utils.data import DataLoader
from models.aae import D_net_gauss, Q_net, P_net
from models.KT_data_loader import AAE_KTDATA
from models.preprocessor import preprocess_features, normalize_features
from database.logger import logger, timed, formatter
from models.custom_train_func import CosineAnnealingWarmUpRestarts, EarlyStopping
from torchviz import make_dot

@timed
def trainer(gpu, args, train_data, ru, logger):
    # refactoring
    train_data = preprocess_features(args, train_data)
    print("available device", torch.cuda.device_count())
    print("current ", gpu)
    
    n_gpu = args['MODEL']['num_gpu']
    uncertainty_check = args['MODEL']['uncertainty_check']
    
    if train_data is None:
        logger.info(f'There is no train data satisfied in processing {ru}')
    else:
        train_data, _ = normalize_features(args['MODEL'], ru, train_data, istrain=True)
        ktdata = AAE_KTDATA(train_data.to_numpy())

        if len(ktdata) >= 100:
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='tcp://127.0.0.1:3456',
                world_size=n_gpu,
                rank=gpu)

            torch.cuda.set_device(gpu)
            all_len = train_data.shape[0]
            train_len = int(all_len * 0.9)
            ktdata_train = ktdata[:train_len]
            ktdata_val = ktdata[train_len:]
            
            batch_szie = int(256 / n_gpu )
            num_worker = int(2 / n_gpu )
            
            train_sampler = torch.utils.data.distributed.DistributedSampler(ktdata_train)
            val_sampler = torch.utils.data.distributed.DistributedSampler(ktdata_val)
            
            train_data = DataLoader(ktdata_train, batch_size = batch_szie, sampler = train_sampler,pin_memory=True, num_workers=num_worker)
            val_data = DataLoader(ktdata_val, batch_size = batch_szie, shuffle= False, sampler = val_sampler,pin_memory=True, num_workers=num_worker)


        else:
            train_data = DataLoader(ktdata, batch_size=32, shuffle=True)
            val_data = DataLoader(ktdata, batch_size=len(ktdata), shuffle=False)
        
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        EPS = 1e-15
        z_red_dims = 20
        Q = Q_net(ktdata[0].shape[0], 42, z_red_dims)
        P = P_net(ktdata[0].shape[0], 42, z_red_dims)
        P_Q = nn.Sequential(Q,P)
        D_gauss = D_net_gauss(42, z_red_dims)
        torch.cuda.set_device(gpu)
        
        Q.cuda(gpu)
        P.cuda(gpu)
        D_gauss.cuda(gpu)
        P_Q.cuda(gpu)
                
                
        Q = torch.nn.parallel.DistributedDataParallel(Q, device_ids=[gpu])
        P = torch.nn.parallel.DistributedDataParallel(P, device_ids=[gpu])
        P_Q = torch.nn.parallel.DistributedDataParallel(P_Q, device_ids=[gpu])
        D_gauss = torch.nn.parallel.DistributedDataParallel(D_gauss, device_ids=[gpu])
        
        # Set learning rates
        gen_lr      = 0.001
        reg_lr      = 0.00005
        
        patience = 5
        early_stopping = EarlyStopping(patience = patience, verbose = True)

        # encode/decode optimizers
        #optim_P_Q   = torch.optim.Adam(list(Q.parameters())+list(P.parameters()),lr=gen_lr)
        optim_Q_en = torch.optim.Adam(Q.parameters(), lr=gen_lr)
        optim_P = torch.optim.Adam(P.parameters(), lr=gen_lr)
#
        # regularizing optimizers
        optim_Q_gen = torch.optim.Adam(Q.parameters(), lr=reg_lr)
        optim_D     = torch.optim.Adam(D_gauss.parameters(), lr=reg_lr)
        



        iter_per_epoch  = len(train_data)
        total_step      = len(train_data) * 100
        print("total_training_step : ",total_step)

        data_iter       = iter(train_data)

        # Start training
        z_real_list     = []
        z_fake_list     = []

        if not os.path.exists(args['MODEL']['aae_state_savepath']):
            os.mkdir(args['MODEL']['aae_state_savepath'])

        if not os.path.exists(args['MODEL']['aae_maha_savepath']):
            os.mkdir(args['MODEL']['aae_maha_savepath'])


        best_mse        = float('inf')
        mse_losses    = nn.MSELoss().to(gpu)
                    
        kl_weight   = 0.2
        kl_loss     = bnn.BKLLoss(reduction='mean',last_layer_only = False).to(gpu)
        
        for step in range(total_step):

            # Reset the data_iter
            if (step+1) % iter_per_epoch == 0:
                data_iter = iter(train_data)
            # Fetch the images and labels and convert them to variables
            images      = next(data_iter)
            images      = images.to(gpu)
            images      = images.float()

            #reconstruction loss
            P.zero_grad()
            Q.zero_grad()
            D_gauss.zero_grad()

            z_sample    = Q(images)   #encode to z
            X_sample    = P(z_sample) #decode to X reconstruction

            mse_loss=mse_losses(X_sample + EPS, images + EPS)
            kl_loss_p_q   = kl_loss(P_Q)
            
  
            recon_loss  = mse_loss + kl_weight*kl_loss_p_q

            recon_loss.backward()
            optim_Q_en.step()
            optim_P.step()            

            # Discriminator
            ## true prior is random normal (randn)
            ## this is constraining the Z-projection to be normal!
            
            Q.eval()
            z_real_gauss    = (torch.randn(images.size()[0], z_red_dims) * 5.).to(gpu)
            if step >= (total_step - iter_per_epoch - 1):
                z_real_list.append(z_real_gauss)
            D_real_gauss    = D_gauss(z_real_gauss)

            z_fake_gauss    = Q(images)
            if step >= (total_step - iter_per_epoch - 1):
                z_fake_list.append(z_fake_gauss)
            D_fake_gauss    = D_gauss(z_fake_gauss)
    
            D_loss          = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))

            D_loss.backward()
            optim_D.step()

            # Generator
            Q.train()
            z_fake_gauss    = Q(images)
            D_fake_gauss    = D_gauss(z_fake_gauss)

            G_loss          = -torch.mean(torch.log(D_fake_gauss + EPS))

            G_loss.backward()
            optim_Q_gen.step()



            if (step) % 100 == 0:

                print('+-----------------------+----------------------------------------+')
                print('|   training step: {}   |     MSE_loss     of P-Q_net: {:.5f}. |'.format(str(step + 1), mse_loss.item()))
                print('|                       |     KL_loss      of P-Q_net: {:.5f}.  |'.format( kl_loss_p_q.item()))


                with torch.no_grad():
                    images  = iter(val_data).__next__()
                    images  = images.to(gpu)
		
                    #images  = images.cuda(gpu)
                    images  = images.float()
                    # reconstruction loss
                    P.zero_grad()
                    Q.zero_grad()
                    D_gauss.zero_grad()
                    P_Q.zero_grad()

                    z_sample    = Q(images)  # encode to z
                    X_sample    = P(z_sample)  # decode to X reconstruction

                    val_mse_loss    = mse_losses(X_sample + EPS, images + EPS)
                    val_kl_loss_p_q   = kl_loss(P_Q)

                    val_recon_loss  = mse_loss + kl_weight*val_kl_loss_p_q
 
                    if val_recon_loss < best_mse:
                        best_mse = val_recon_loss
                        print('+-----------------------+----------------------------------------+')
                        print('| save model at step:{} |  Valid_MSE_loss  of P-Q_net: {:.5f}. |'.format(str(step + 1), val_mse_loss.item()))
                        print('|                       |  Valid_KL_loss   of P-Q_net: {:.5f}.  |'.format(val_kl_loss_p_q.item()))
                        print('+----------------------------------------------------------------+')
                        torch.save(Q.state_dict(), os.path.join(args['MODEL']['aae_state_savepath'], f'Q_{ru}.ckpt'))
                        torch.save(P.state_dict(), os.path.join(args['MODEL']['aae_state_savepath'], f'P_{ru}.ckpt'))
                        
                        test_data   = DataLoader(ktdata, batch_size=len(ktdata), shuffle=False)
                        a           = iter(test_data).__next__()
                        z_sample    = Q(a.to(gpu).float())
                        a           = pd.DataFrame(z_sample.detach().cpu().numpy())

                        robust_cov  = MinCovDet().fit(a)

                        with open(os.path.join(args['MODEL']['aae_maha_savepath'], f'{ru}.pickle'), "wb") as f:
                            pickle.dump(robust_cov, f)
                        #print('save cov matrix')
            
                early_stopping(val_recon_loss, P_Q) ## save_checkpoint 주석처리
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                            
            ### :
        if uncertainty_check == args['MODEL']['uncertainty_checker_train'] :
            print("============================================")
            print("Start to Check a Uncertainty for Current AAE")
            print(":")
#                uncertainty_for_z = []
            uncertainty_for_x_hat = []
#                tmp_uncertainty = []
            ### load best model

            with torch.no_grad():
                images  = iter(val_data).__next__()
                images  = images.to(gpu)
#                    images  = images.cuda()
                images  = images.float()
                # reconstruction loss
                P.zero_grad()
                Q.zero_grad()
                D_gauss.zero_grad()

                Q.load_state_dict(torch.load(os.path.join(args['MODEL']['aae_state_savepath'], f"Q_{ru}.ckpt"),map_location=f'cuda:{gpu}'))
                P.load_state_dict(torch.load(os.path.join(args['MODEL']['aae_state_savepath'], f"P_{ru}.ckpt"),map_location=f'cuda:{gpu}'))
                for i in range(0,50):


                    z_sample    = Q(images)  # encode to z
                    X_sample    = P(z_sample)  # decode to X reconstruction
                    
                    val_mse_loss    = mse_losses(X_sample + EPS, images + EPS)
                
                    uncertainty_for_x_hat.append(val_mse_loss.item())

#                    print(f"step {i}: the shape of uncertainty_for_z_vector : ", shape(uncertainty_for_z))
                #print(f"step {i}: uncertainty_for_X_sample : ", (uncertainty_for_x_hat))

            
            #### Uncertainty Value Check
            print('+--------------------------------+-------------------------------+')
            print('|    Uncertainty for Testset:    |     mean of z  : {:.5f}.      |'.format( np.mean(uncertainty_for_x_hat,axis=0)))
            print('|            z_vector            |      var of z  : {:.5f}.      |'.format( np.var(uncertainty_for_x_hat,axis=0)))
            print('+--------------------------------+-------------------------------+')
            
            
            if not os.path.exists(args['MODEL']['train_uncertaintiy_savepath']):
                os.mkdir(args['MODEL']['train_uncertaintiy_savepath'])
            
            if not os.path.exists(os.path.join(args['MODEL']['train_uncertaintiy_savepath'], f'{ru}.pkl')):
                with open(os.path.join(args['MODEL']['train_uncertaintiy_savepath'], f'{ru}.pkl'), 'wb') as f:
                    pickle.dump(uncertainty_for_x_hat,f)
                print ("save the first train_uncertainty")
            else:
                with open(os.path.join(args['MODEL']['train_uncertaintiy_savepath'],f'{ru}.pkl'), 'rb') as f:
                    print(f)
                    data = pickle.load(f)
                data = np.append(data,uncertainty_for_x_hat)
                with open(os.path.join(args['MODEL']['train_uncertaintiy_savepath'], f'{ru}.pkl'), 'wb') as f:
                    pickle.dump(data,f)
            
            ## save uncertayiyny
            print(":")
            print("======================================================")
        ## :
#        torch.save(Q.state_dict(), os.path.join(args['aae_state_savepath'], f'{ru}_final.ckpt'))
#        torch.save(P.state_dict(), os.path.join(args['aae_state_savepath'], f'{ru}_final.ckpt'))

        with torch.no_grad():
            test_data   = DataLoader(ktdata, batch_size=len(ktdata), shuffle=False)
	
            
            a           = iter(test_data).__next__()

            z_sample    = Q(a.to(gpu).float())

            a           = pd.DataFrame(z_sample.detach().cpu().numpy())
            robust_cov  = MinCovDet().fit(a)

        with open(os.path.join(args['MODEL']['aae_maha_savepath'], f'{ru}_final.pickle'), "wb") as f:
            pickle.dump(robust_cov, f)


if __name__ == '__main__':
    ru_list = []
    data_paths = glob.glob('../data/temp/training/*-*-*_*:*:*.csv')
    data_paths.sort()
    data_path = data_paths[-1]
    for ru in ru_list:
        train_data = pd.read_csv(data_path)
        train_data = train_data[(train_data['ru_id'] == ru)]
        trainer(train_data, None)




