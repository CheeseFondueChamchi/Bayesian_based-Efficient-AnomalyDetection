import sys
sys.path.append('..')
import os
import glob
import horovod.torch as hvd
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
from model.custom_train_func import CosineAnnealingWarmUpRestarts, EarlyStopping

@timed
def trainer(args, train_data, ru, logger):
    # refactoring
    train_data = preprocess_features(args, train_data)
    print("available device", torch.cuda.device_count())
    print("current ", torch.cuda.current_device())

    hvd.init()   # horovod
    torch.cuda.set_device(hvd.local_rank()) #horovod
    torch.set_num_threads(1)

    if train_data is None:
        logger.info(f'There is no train data satisfied in processing {ru}')
    else:
        train_data, _ = normalize_features(args, ru, train_data, istrain=True)
        ktdata = AAE_KTDATA(train_data.to_numpy())

        if len(ktdata) >= 100:
            all_len = train_data.shape[0]
            train_len = int(all_len * 0.9)
            ktdata_train = ktdata[:train_len]
            ktdata_val = ktdata[train_len:]
            train_sampler = torch.utils.data.distributed.DistributedSampler(ktdata_train, num_replicas = hvd.size(), rank = hvd.rank())
            train_data = DataLoader(ktdata_train, batch_size = 32, sampler = train_sampler)
            val_sampler = torch.utils.data.distributed.DistributedSampler(ktdata_val, num_replicas = hvd.size(), rank = hvd.rank())
            val_data = DataLoader(ktdata_val, batch_size = 32, shuffle= False, sampler = val_sampler)


        else:
            train_data = DataLoader(ktdata, batch_size=32, shuffle=True)
            val_data = DataLoader(ktdata, batch_size=len(ktdata), shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        EPS = 1e-15
        z_red_dims = 20
        Q = Q_net(ktdata[0].shape[0], 32, z_red_dims)
        P = P_net(ktdata[0].shape[0], 32, z_red_dims)
        D_gauss = D_net_gauss(32, z_red_dims)
        
        Q.cuda()
        P.cuda()
        D_gauss.cuda()

        # Set learning rates
        gen_lr      = 0.0001
        reg_lr      = 0.00005
        
        patience = 25
        early_stopping = EarlyStopping(patience = patience, verbose = True)

        # encode/decode optimizers
        optim_P_Q   = torch.optim.Adam(list(P.parameters()) + list(Q.parameters()),lr=gen_lr)
#        optim_P = torch.optim.Adam(P.parameters(), lr=gen_lr)
#        optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=gen_lr)
        # regularizing optimizers
        optim_Q_gen = torch.optim.Adam(Q.parameters(), lr=reg_lr)
        optim_D     = torch.optim.Adam(D_gauss.parameters(), lr=reg_lr)
        
        hvd.broadcast_parameters(P.state_dict(), root_rank=0) #horovod
        hvd.broadcast_parameters(Q.state_dict(), root_rank = 0)
        hvd.broadcast_parameters(D_gauss.state_dict(), root_rank = 0) 


        iter_per_epoch  = len(train_data)
        total_step      = len(train_data) * 10

        data_iter       = iter(train_data)

        # Start training
        z_real_list     = []
        z_fake_list     = []

        if not os.path.exists(args['aae_state_savepath']):
            os.mkdir(args['aae_state_savepath'])

        if not os.path.exists(args['aae_maha_savepath']):
            os.mkdir(args['aae_maha_savepath'])


        best_mse        = float('inf')
        
        for step in range(total_step):

            # Reset the data_iter
            if (step+1) % iter_per_epoch == 0:
                data_iter = iter(train_data)
            # Fetch the images and labels and convert them to variables
            images      = next(data_iter)
            images      = images.to(device)
            images      = images.float()

            #reconstruction loss
            P.zero_grad()
            Q.zero_grad()
            D_gauss.zero_grad()

            z_sample    = Q(images)   #encode to z
            X_sample    = P(z_sample) #decode to X reconstruction
            P_Q         = nn.Sequential(Q,P)
            
            mse_loss    = nn.MSELoss()(X_sample + EPS, images + EPS)

            kl_loss     = bnn.BKLLoss(reduction='mean',last_layer_only = False)
            kl_weight   = 0.1
            kl_loss_p_q   = kl_loss(P_Q)
            print("mse_loss :",mse_loss)
            print("kl_loss_p_q :", kl_loss_p_q)
#            print("kl_loss_p:", kl_loss_p)
#            print("kl_loss_q:", kl_loss_p)

            
            recon_loss  = mse_loss + kl_weight*kl_loss_p_q
#            recon_loss  = mse_loss + kl_weight*kl_loss_p_q + kl_weight*kl_

            recon_loss.backward()
            optim_P_Q.step()
#            optim_P.step()
#            optim_Q_enc.step()

            # Discriminator
            ## true prior is random normal (randn)
            ## this is constraining the Z-projection to be normal!
            
            Q.eval()
            z_real_gauss    = (torch.randn(images.size()[0], z_red_dims) * 5.).to(device)
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
                print('|----------------------------------------|')
                print('|step: {} |  MSE_loss  of P-Q_net: {:.3f}|'.format(str(step + 1), mse_loss.item()))
                print('|         |   KL_loss  of P-Q_net: {:.3f}|'.format(str(step + 1), kl_loss_p_q.item()))
                print('|----------------------------------------|')

            if (step) % 10 == 0:
                with torch.no_grad():
                    images  = iter(val_data).__next__()
                    images  = images.to(device)
                    images  = images.cuda()
                    images  = images.float()
                    # reconstruction loss
                    P.zero_grad()
                    Q.zero_grad()
                    D_gauss.zero_grad()

                    z_sample    = Q(images)  # encode to z
                    X_sample    = P(z_sample)  # decode to X reconstruction

                    val_mse_loss    = nn.MSELoss()(X_sample + EPS, images + EPS)
                    
                    P_Q         = nn.Sequential(Q,P)
                    
                    val_kl_loss     = bnn.BKLLoss(reduction='mean',last_layer_only = False)
                    val_kl_weight   = 0.1
                    val_kl_loss_p_q   = val_kl_loss(P_Q)

                    val_recon_loss  = mse_loss + val_kl_weight*val_kl_loss_p_q
 
                    if val_recon_loss < best_mse:
                        best_mse = val_recon_loss
                        print('|------------------------------------------------------|')
                        print('|save model at step: {} |  MSE_loss  of P-Q_net: {:.3f}|'.format(str(step + 1), val_mse_loss.item()))
                        print('|                       |   KL_loss  of P-Q_net: {:.3f}|'.format(str(step + 1), val_kl_loss_p_q.item()))
                        print('|------------------------------------------------------|')
                        torch.save(Q.state_dict(), os.path.join(args['aae_state_savepath'], f'Q_{ru}.ckpt'))
                        torch.save(P.state_dict(), os.path.join(args['aae_state_savepath'], f'P_{ru}.ckpt'))
                        
                        test_data   = DataLoader(ktdata, batch_size=len(ktdata), shuffle=False)
                        a           = iter(test_data).__next__()
                        z_sample    = Q(a.to(device).float())
                        a           = pd.DataFrame(z_sample.detach().cpu().numpy())
                        robust_cov  = MinCovDet().fit(a)

                        with open(os.path.join(args['aae_maha_savepath'], f'{ru}.pickle'), "wb") as f:
                            pickle.dump(robust_cov, f)
                        print('save cov matrix')
            
                early_stopping(val_recon_loss, P_Q) ## save_checkpoint 주석처리
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                            
            ### :
            if uncertainty_check == args['uncertainty_checker_train'] :
                print("============================================")
                print("Start to Check a Uncertainty for Current AAE")
                print(":")
#                uncertainty_for_z = []
                uncertainty_for_x_hat = []
#                tmp_uncertainty = []
                ### load best model

                with torch.no_grad():
                    images  = iter(val_data).__next__()
                    images  = images.to(device)
                    images  = images.cuda()
                    images  = images.float()
                    # reconstruction loss
                    P.zero_grad()
                    Q.zero_grad()
                    D_gauss.zero_grad()

                    
                    for i in range(0,100):
                        Q.load_state_dict(torch.load(os.path.join(model_conf['aae_state_savepath'], f"Q_{ru}.ckpt"),map_location=device))
                        P.load_state_dict(torch.load(os.path.join(model_conf['aae_state_savepath'], f"P_{ru}.ckpt"),map_location=device))
                        z_sample    = Q(images)  # encode to z
                        X_sample    = P(z_sample)  # decode to X reconstruction
                        
                        val_mse_loss    = nn.MSELoss()(X_sample + EPS, images + EPS)
                    
                        uncertainty_for_x_hat.append(val_mse_loss.item())

#                    print(f"step {i}: the shape of uncertainty_for_z_vector : ", shape(uncertainty_for_z))
                    print(f"step {i}: the shape of uncertainty_for_X_sample : ", shape(uncertainty_for_x_hat))
                    print(":")
 
                
                #### Uncertainty Value Check
                print('|-------------------------------------------------|')
                print('| Uncertainty for Testset: |   mean of z  : {:.3f}|'.format( np.mean(uncertainty_for_x_hat,axis=0)))
                print('|         z_vector         |    var of z  : {:.3f}|'.format( np.var(uncertainty_for_x_hat=0)))
                print('|-------------------------------------------------|')
                
                
                if not os.path.exists(model_conf['train_uncertaintiy_savepath']):
                    os.mkdir(model_conf['train_uncertaintiy_savepath'])
                
                if not os.path.exists(os.path.join(model_conf['train_uncertaintiy_savepath'], f'{ru}.pkl')):
                    with open(os.path.join(model_conf['train_uncertaintiy_savepath'], f'{ru}.pkl'), 'wb') as f:
                        pickle.dump(loss_save,f)
                    print ("save the first train_uncertainty")
                else:
                    with open(os.path.join(model_conf['train_uncertaintiy_savepath'],f'{ru}.pkl'), 'rb') as f:
                        data = pickle.load(f)
                    data = np.append(data,loss_save)
                    with open(os.path.join(model_conf['train_uncertaintiy_savepath'], f'{ru}.pkl'), 'wb') as f:
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
            z_sample    = Q(a.to(device).float())
            a           = pd.DataFrame(z_sample.detach().cpu().numpy())
            robust_cov  = MinCovDet().fit(a)

        with open(os.path.join(args['aae_maha_savepath'], f'{ru}_final.pickle'), "wb") as f:
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




