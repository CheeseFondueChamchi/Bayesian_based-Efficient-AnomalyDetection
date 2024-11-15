import sys
sys.path.append('..')
sys.path.append('/home/infra/project/KT-AD-2021/')
import glob
import time
import os
import torch
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)

from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from models.aae import Q_net, P_net
from models.KT_data_loader import AAE_KTDATA
from models.preprocessor import preprocess_features, normalize_features
from models.z_score import calc_z_score

from database.connection import ConnManager
from database.sqls import get_insert_inference_output_sql, get_update_flag_sql, update_complete_time
from database.logger import logger, timed, formatter


@timed
def evaluator_mp(args, ori_data, ru_list,uncertainty_checker=False):
    model_conf = args['MODEL']
    db_conf = args['DB']
    xgb_cl = xgb.XGBClassifier()
    xgb_cl.load_model("xgb_model_1202.bin")
    ###0 0929 Label update

    label_info = ['Coverage 초과', 'RU Aging','LTE-NR 네이버 오류','트래픽 패턴 변화','Heavy User/속도 측정','RSSI 패턴 변화']
    act_info = [ '최적화/증설 검토','RU 장비 리셋','LTE-NR 네이버 재점검','트래픽 추이 모니터링','자동 복구 여부 확인','중계기 점검']
    
    for ru in ru_list:
        ru = str(ru)
        if os.path.exists(os.path.join(model_conf['aae_maha_savepath'], f"{ru}.pickle")):
            db_dict = {}
            logger.info(f"inference for {ru}")
            with open(os.path.join(model_conf['aae_maha_savepath'], f"{ru}.pickle"), "rb") as f:
                robust_cov = pickle.load(f)

            eval_data = ori_data[ori_data['ru_id'] == ru].copy()

            db_dict['datetime'] = eval_data['datetime'].tolist()
            db_dict['du_id'] = eval_data['du_id'].to_numpy().tolist()
            db_dict['cell_id'] = eval_data['cell_id'].to_numpy().astype('str').astype('object').tolist()
            db_dict['gnb_id'] = eval_data['gnb_id'].to_numpy().tolist()
            db_dict['ru_id'] = eval_data['ru_id'].to_numpy().tolist()
            
            ###1 0929 for RCA conditions
            v_RachPA=eval_data['rachpreamblea']
            prb_dlavg=eval_data['totprbdlavg']
            scg_f_ratio = eval_data['scgfailratio']
            endc_rate = eval_data['endcchgrate']
            endc_addrate = eval_data['endcaddrate']
            scgfailoper_dl_count = eval_data['scgfailoper_dl_t310expiry_count']
            airmac_ul_dl_byte = eval_data['airmacdlbyte']+eval_data['airmaculbyte']


            # drop and making features
            eval_data = preprocess_features(model_conf, eval_data)
            if eval_data is None:
                logger.info(f'There is no eval data satisfied in processing {ru}')
                continue

            # new score
            key_endc_add_att = eval_data['endcaddatt'].to_numpy()
            key_endc_add_att[key_endc_add_att <= 9] = 0
            key_endc_add_att[(key_endc_add_att > 9) & (key_endc_add_att <= 29)] = 1
            key_endc_add_att[key_endc_add_att > 29] = 2

            if len(eval_data) == 0:
                logger.info(f'There is no observation for evaluating in {ru}')
                pass
            else:
                calc_z_data = eval_data.copy()

                eval_data, te = normalize_features(model_conf, ru,  eval_data, istrain=False)

                # calculating z score for this cell
                # z_sum, f_list, z_list, f_tot, z_tot, cls_feature = calc_z_score(calc_z_data, te)
                
                # calculating z score for this cell
                # 210928 z_score test: spare0 add
                z_sum, f_list, z_list, f_tot, z_tot, cls_feature, spare0, spare1 = calc_z_score(calc_z_data, te)

                ktdata = AAE_KTDATA(eval_data.to_numpy())
                test_data = DataLoader(ktdata, batch_size=len(ktdata), shuffle=False)
                device = torch.device("cpu")

                z_red_dims = 20
                Q = Q_net(eval_data.shape[1], 32, z_red_dims).to(device)
                P = P_net(eval_data.shape[1], 32, z_red_dims).to(device)

                
                #########################
                if uncertainty_check == True :
                    print("============================================")
                    print("Start to Check a Uncertainty for Current AAE")
                    print(":")
    #                uncertainty_for_z = []
                    uncertainty_for_x_hat = []
    #                tmp_uncertainty = []
                    ### load best model

                    with torch.no_grad():
                        images  = iter(test_data).__next__()
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
                    print('| Uncertainty for Validset:|   mean of z  : {:.3f}|'.format( np.mean(uncertainty_for_x_hat,axis=0)))
                    print('|         z_vector         |    std of z  : {:.3f}|'.format( np.std(uncertainty_for_x_hat=0)))
                    print('|-------------------------------------------------|')
                    
                    
                    if not os.path.exists(model_conf['valid_uncertaintiy_savepath']):
                        os.mkdir(model_conf['valid_uncertaintiy_savepath'])
                    
                    if not os.path.exists(os.path.join(model_conf['valid_uncertaintiy_savepath'], f'{ru}.pkl')):
                        with open(os.path.join(model_conf['valid_uncertaintiy_savepath'], f'{ru}.pkl'), 'wb') as f:
                            pickle.dump(uncertainty_for_x_hat,f)
                        print ("save the first valid_uncertainty")
                    else:
                        with open(os.path.join(model_conf['valid_uncertaintiy_savepath'],f'{ru}.pkl'), 'rb') as f:
                            data = pickle.load(f)
                        data = np.append(data,uncertainty_for_x_hat)
                        with open(os.path.join(model_conf['valid_uncertaintiy_savepath'], f'{ru}.pkl'), 'wb') as f:
                            pickle.dump(data,f)
                ##################
                
                
                Q.load_state_dict(torch.load(os.path.join(model_conf['aae_state_savepath'], f"Q_{ru}.ckpt"),
                                             map_location=device))
                Q.to(device)
                Q.eval()

                a = iter(test_data).__next__()
                z_sample = Q(a.to(device).float())
                a = pd.DataFrame(z_sample.detach().cpu().numpy())
                
                maha = robust_cov.mahalanobis(a)

                # anomaly score
                # score0 = 20 * np.log10(maha)
                score0 = 33 * np.log10(maha)


                ###2 0~100 Scaler
                score0[score0 >= 80] = 100 - (100 - 20) * ((80 / score0[score0 >= 80]) ** 3)
                thres = np.array(100)
                spare1 = spare1.to_numpy()
                score0[key_endc_add_att == 0] = score0[key_endc_add_att == 0] * 0.4
                score0[key_endc_add_att == 1] = score0[key_endc_add_att == 1] * 0.8
                # TODO: Check 0802
                spare2 = score0[0] * (z_sum/120)**0.5

            

                # score0[z_sum < 45] = score0[z_sum < 45] * 0.5
                db_dict['score0'] = score0.tolist()
                db_dict['z_sum'] = z_sum.tolist()
                db_dict['f_list'] = f_list.tolist()  # top feature name
                db_dict['z_list'] = z_list.tolist()  # top feature value
                db_dict['f_tot'] = f_tot.tolist()    # all feature name
                db_dict['z_tot'] = z_tot.values.tolist()    # all feature value

                
                
                ###3 210928 z_score test: spare0 add
                db_dict['spare0'] = spare0.tolist() # z_sum value test
                db_dict['spare1'] = spare1.tolist()  # z_sum_norm value test
                db_dict['spare2'] = spare2.tolist() # Anomaly score test

                # TODO: Check 0802
                db_dict['rca_num'], db_dict['rca_str'], db_dict['act_num'], \
                db_dict['act_str'], db_dict['score1'] = [], [], [], [], []
                
                
                
                ###4 20211013 RCA Rule 추가
                zero_call_condition = airmac_ul_dl_byte.values[0] == 0
                
                endc_condition = cls_feature['endcaddrate'].values[0]<-40 or (cls_feature['endcrelbymenb'].values[0] > 10 and endc_addrate.values[0] <= 80)
                
                traffic_heavy_user_condition =scg_f_ratio.values[0]<30 and cls_feature['rssipathavg'].values[0]< 15
                traffic_pattern_condition = cls_feature['totprbulavg'].values[0]>30 or cls_feature['totprbdlavg'].values[0]>30 or cls_feature['uenoavg'].values[0]>5
                heavy_user_condition = cls_feature['totprbdlavg'].values[0] > 20 and prb_dlavg.values[0] > 90 #
                
            
                ul_retr_increse_condition = cls_feature['scgfailoper_ul_rlcmaxnumretx_count'].values[0]>10
                
                dl_retr_increse_condition = cls_feature['scgfailoper_dl_rlcmaxnumretx_count'].values[0]>10
                
                lte_nr_nav_condition =cls_feature['scgfail'].values[0] > 20 and cls_feature['rssipathavg'].values[0]<=5 and cls_feature['bler_ul'].values[0]<=5 and cls_feature['bler_dl'].values[0]<=5 and cls_feature['rachpreamblea'].values[0]<5 and scg_f_ratio.values[0] > 40
                
#                nr_nr_nav_condition = cls_feature['endcchgrate'].values[0]<-5 and cls_feature['rssipathavg'].values[0] < 10 and endc_rate.values[0] >5
                nr_nr_nav_condition = cls_feature['scgfailoper_dl_t310expiry_count'].values[0] > 3.5 and endc_rate.values[0] < 50 and scgfailoper_dl_count.values[0] > 10
                
                ru_aging_condition = cls_feature['scgfail'].values[0] >20 and cls_feature['rssipathavg'].values[0] >=10 and cls_feature['bler_ul'].values[0] > 3 and cls_feature['bler_dl'].values[0] > 3 and cls_feature['rachpreamblea'].values[0]<5
                

                coverage_sub_condition = cls_feature['bler_ul'].values[0] <= 6 or cls_feature['bler_dl'].values[0] <= 6
                coverage_exceed_condition = cls_feature['scgfail'].values[0] >10 and cls_feature['rssipathavg'].values[0] < 20 and coverage_sub_condition and cls_feature['rachpreamblea'].values[0] >= 5
                
                rssi_pattern_condition = scg_f_ratio.values[0] < 10 and cls_feature['scgfail'].values[0]<3 and cls_feature['uenoavg'].values[0]<5 and cls_feature['rssipathavg'].values[0]>30 and cls_feature['bler_ul'].values[0]<3 and cls_feature['bler_dl'].values[0]<3
                

                for idx, score in enumerate(score0):

                    if score >= 60:
                        # 1013 rachpreamble == 0 -> Zero Call fix
                        ###5 RCA_rule classifier
                        if zero_call_condition :
                            # db_dict['rca_num'].append(0)
                            # db_dict['rca_str'].append('rca_tmp')

 
                            db_dict['rca_str'].append("Zero Call 발생")
                            db_dict['rca_num'].append(11)
                            db_dict['act_num'].append(11)
                            db_dict['act_str'].append("RU 또는 DSP 리셋")
                            db_dict['score1'].append(1)
                        elif endc_condition :

 
                            db_dict['rca_str'].append("ENDC 문제")
                            db_dict['rca_num'].append(9)
                            db_dict['act_num'].append(9)
                            db_dict['act_str'].append("LTE RU 또는 특정 단말 문제")
                            db_dict['score1'].append(1)
                            
                        elif traffic_heavy_user_condition:
                            if traffic_pattern_condition:

     
                                db_dict['rca_str'].append("트래픽 패턴 변화")
                                db_dict['rca_num'].append(4)
                                db_dict['act_num'].append(4)
                                db_dict['act_str'].append("트래픽 추이 모니터링")
                                db_dict['score1'].append(1)
                            elif heavy_user_condition:
                                db_dict['rca_str'].append("Heavy User/속도 측정")
                                db_dict['rca_num'].append(5)
                                db_dict['act_num'].append(5)
                                db_dict['act_str'].append("자동 복구 여부 확인")
                                db_dict['score1'].append(1)
                                
                        elif ul_retr_increse_condition :

 
                            db_dict['rca_str'].append("UL 재전송 증가")
                            db_dict['rca_num'].append(7)
                            db_dict['act_num'].append(7)
                            db_dict['act_str'].append("채널카드 리셋")
                            db_dict['score1'].append(1)
                        elif dl_retr_increse_condition :

 
                            db_dict['rca_str'].append("DL 재전송 증가")
                            db_dict['rca_num'].append(8)
                            db_dict['act_num'].append(8)
                            db_dict['act_str'].append("DSP 리셋")
                            db_dict['score1'].append(1)
                        elif lte_nr_nav_condition :

 
                            db_dict['rca_str'].append("LTE-NR 네이버 오류")
                            db_dict['rca_num'].append(3)
                            db_dict['act_num'].append(3)
                            db_dict['act_str'].append("LTE-NR 네이버 재점검")
                            db_dict['score1'].append(1)
 
                        elif nr_nr_nav_condition :
                            db_dict['rca_str'].append("NR-NR 네이버 오류")
                            db_dict['rca_num'].append(10)
                            db_dict['act_num'].append(10)
                            db_dict['act_str'].append("NR-NR 네이버 점검 및 위⋅경도 확인")
                            db_dict['score1'].append(1)
 
                        elif ru_aging_condition :
                            db_dict['rca_str'].append("RU Aging")
                            db_dict['rca_num'].append(2)
                            db_dict['act_num'].append(2)
                            db_dict['act_str'].append("RU 장비 리셋")
                            db_dict['score1'].append(1)
 
 
 
                        elif coverage_exceed_condition :
                            db_dict['rca_str'].append("Coverage 초과")
                            db_dict['rca_num'].append(1)
                            db_dict['act_num'].append(1)
                            db_dict['act_str'].append("최적화/증설 검토")
                            db_dict['score1'].append(1)
 
                        elif rssi_pattern_condition :
                            db_dict['rca_str'].append("RSSI 패턴 변화")
                            db_dict['rca_num'].append(6)
                            db_dict['act_num'].append(6)
                            db_dict['act_str'].append("중계기 점검")
                            db_dict['score1'].append(1)
                            
                        else:
                            
                            cls_feature_temp = cls_feature.copy()
                            ###7
                            cls_feature_temp = cls_feature_temp.drop(['endcrelbymenb'],axis=1)
                            
                            pred_cls = xgb_cl.predict(cls_feature_temp.iloc[idx:(idx + 1)])[0]
 
                            db_dict['rca_str'].append(label_info[pred_cls])
                            db_dict['rca_num'].append(int(pred_cls)+1)
                            db_dict['act_num'].append(int(pred_cls)+1)
                            db_dict['act_str'].append(act_info[pred_cls])
                            db_dict['score1'].append(xgb_cl.predict_proba(cls_feature_temp.iloc[idx:(idx+1)]).max())

                    else:
                        db_dict['rca_num'].append(None)
                        db_dict['rca_str'].append(None)
                        db_dict['act_num'].append(None)
                        db_dict['act_str'].append(None)
                        db_dict['score1'].append(None)

                # ---------------------------------------------------#
                conn_manager = ConnManager(db_conf)
                # conn_string = "host='172.21.222.198' port='5432' dbname='bdo' user='bdo_ai' password='bdo1234!@'"
                conn = conn_manager.conn
                cur = conn.cursor()

                # datetime, du_id, cell_id, score0, z_sum  shape:(n, 5)
                
                
                ###3 210928 z_score test: spare0 add
                for datetime, ru_id, du_id, cell_id, score0, score1, z_sum, f_list, z_list, f_tot, z_tot, \
                    rca_num, rca_str, act_num, act_str, spare0, spare1, spare2 in \
                    zip(db_dict['datetime'], db_dict['ru_id'], db_dict['du_id'], db_dict['cell_id'],
                        db_dict['score0'], db_dict['score1'],
                        db_dict['z_sum'], db_dict['f_list'], db_dict['z_list'], db_dict['f_tot'], db_dict['z_tot'],
                        db_dict['rca_num'], db_dict['rca_str'], db_dict['act_num'], db_dict['act_str'], db_dict['spare0'], db_dict['spare1'], db_dict['spare2']):

                    f_list = ','.join(f_list)
                    z_list = ','.join([f'{z:.2f}' for z in z_list])
                    f_tot = ','.join(f_tot)
                    z_tot = ','.join([f'{z:.1f}' for z in z_tot])[:300]

                    # insert query
                    ###3 210928 z_score test: spare0 add
                    st = time.time()
                    sql = get_insert_inference_output_sql(datetime=datetime, du_id=du_id, cell_id=cell_id, ru_id=ru_id,
                                                          score0=score0, score1=score1, z_sum=z_sum, f_list=f_list, z_list=z_list,
                                                          f_tot=f_tot, z_tot=z_tot, rca_num=rca_num, rca_str=rca_str,
                                                          act_num=act_num, act_str=act_str,  spare0=spare0, spare1=spare1, spare2=spare2)
                    #print(sql)
                    cur.execute(sql)
                    ed = time.time()
                    logger.info(f'insert sql executed time: {ed-st:.2f}')

                    # update query
                    sql = get_update_flag_sql(datetime=datetime, du_id=du_id, cell_id=cell_id, ru_id=ru_id)
                    # print(sql)
                    cur.execute(sql)
                    ed2 = time.time()
                    logger.info(f'update sql executed time: {ed2-ed:.2f}')

                    conn.commit()

                #cur.execute('select * from tbl_ai_output')
                #print('inserted', cur.fetchall())

                #cur.execute('select * from tbl_stat_5g_cu_du_5min where flag=1')
                #print('updated', cur.fetchall())

                if not os.path.exists("test_results"):
                    os.mkdir("test_results")

                with open("models/test_results/{}_anom_score.pickle".format(ru), "wb") as f:
                    pickle.dump(score0, f)
                with open("models/test_results/{}_z_score.pickle".format(ru), "wb") as f:
                    pickle.dump(z_sum, f)
        else:
            print('In this {} case, there is no statistics for calcluating Mahalanobis distance.'.format(ru))

@timed
def inference_and_insert(conf, logger, data=None):
    logger.info(f'Start with config: {conf}')
    db_conf = conf['DB']
    model_conf = conf['MODEL']
    multi_proc = model_conf['multi_proc']

    if data is None:
        data_paths = glob.glob('./data/temp/inference/*-*-*_*:*:*.csv')
        data_paths.sort()
        data_path = data_paths[-1]
        eval_data = pd.read_csv(data_path)
    else:
        eval_data = data

    ru_list = eval_data['ru_id'].value_counts().index.tolist()
    max_num_of_ru = model_conf['max_cell'] if model_conf['max_cell'] != -1 else len(ru_list)
    ru_list = ru_list[:max_num_of_ru]
    logger.info(f'The number of unique ru: {len(ru_list)}')

    if not multi_proc:
        for ru in ru_list:
            evaluator_mp(conf, eval_data, ru)
    else:
        ###### multi-proc start
        # divide the process for multi-process
        n_procs = model_conf['n_proc']
        cell_chunks = []
        num_chunk = len(ru_list) // n_procs
        rest = len(ru_list) % n_procs

        start = 0
        for i in range(n_procs - 1):
            if i < rest:
                end = start + num_chunk + 1
            else:
                end = start + num_chunk

            cell_chunks.append(ru_list[start:end])
            start = end

        cell_chunks.append(ru_list[start:])

        # making the processes
        processes = []
        for rank in range(n_procs):
            cell_list_current_rank = cell_chunks[rank]
            p = mp.Process(target=evaluator_mp, args=(conf, eval_data, cell_list_current_rank))
            p.start()
            processes.append(p)

        # run processes
        for p in processes:
            p.join()


if __name__ == '__main__':
    import argparse
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'

    parser = argparse.ArgumentParser()
    parser.add_argument('--multi-proc', type=boolean_string, default=True)

    args = parser.parse_args()
    # multi_proc = args.multi_proc
    inference_and_insert(vars(args))
