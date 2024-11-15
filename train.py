import os
import argparse
import yaml
import torch
import pandas as pd
import datetime
import logging
from database.logger import logger, timed, formatter
from database.connection import ConnManager
from database.get_train_data import get_train_data_by_ru, get_ru_unique_list
from models.trainer import trainer


def read_target_csv(path):
    df = pd.read_csv(path, index_col='RU_ID')
    #df['REVISION_FLAG'] = df['FLAG_0']
    df['REVISION_FLAG'] = df['REVISION_FLAG'].fillna(0)
    revision_list = df[df.REVISION_FLAG == 1].index.tolist()
    default_list = df[df.REVISION_FLAG == 0].index.tolist()
    return df, default_list, revision_list


def train_ru(conf, ru_id, ru_ids, duration, isrevision):
    #logger.info(f'[{i}/{len(ru_ids)}] {ru_id} training start')
    if duration is not None:
        st, ed = duration.split('/')
        st = None if len(st) == 0 else st
        ed = None if len(ed) == 0 else ed
    else:
        st, ed = None, None
    
    n_gpu =  torch.cuda.device_count()
    conf['DB']['num_worker'] = n_gpu
    
    conn_manager = ConnManager(conf['DB'])
    train_df = get_train_data_by_ru(conn_manager.conn, ru_id, st=st, ed=ed)
    conn_manager.conn.close()

    cur_state_str = 'Default training'
    
    if isrevision:
        cur_state_str = 'Revision training'
    if len(train_df) == 0 :
        logger.info(f'{cur_state_str} remain[{len(ru_ids)}] {ru_id} for {duration} DataFrame is empty')
        return 0
    else :
        logger.info(f'{cur_state_str} remain[{len(ru_ids)}] {ru_id} for {duration} DataFrame has {len(train_df)} rows')
        ######조건 부 모델 업데이트 ########
        if conf['MODEL']['condition_training']:
            if not os.path.exists(os.path.join(conf['MODEL']['valid_uncertaintiy_savepath'],f'{ru_id}.pkl')):
                ##비교 못함
                torch.multiprocessing.spawn(trainer, nprocs =n_gpu, args=(conf, train_df, ru_id,logger))
                logger.info (f'{cur_state_str} remain[{len(ru_ids)}] training done')
                return 0
            else:
                # cur_loss and valid_loss  비교
                with open(os.path.join(conf['MODEL']['valid_uncertaintiy_savepath'], f"{ru_id}.pkl"), "rb") as f:
                    val_losses = pickle.load(f)
                val_loss_mean = np.mean(val_losses)
                val_loss_std = np.std(val_losses)
#                nomalized_uncertainty = np.abs(val_loss_std - val_loss_mean)/val_loss_mean
                print("==================")
                print("val_loss_mean :",val_loss_mean)
                print(" val_loss_std :",val_loss_std)
#                print("Nomalized_Unce:",nomalized_uncertainty)
                print("==================")
                
                if val_loss_std/val_loss_mean >= 0.8:
                    torch.multiprocessing.spawn(trainer, nprocs =n_gpu, args=(conf, train_df, ru_id,logger))
                    logger.info (f'{cur_state_str} remain[{len(ru_ids)}] training done')
                
                    return 1
                else:
                    logger.info(f"Don't have to train for {ru_id} again")
                    logger.info (f'{cur_state_str} remain[{len(ru_ids)}] training done')
#                    value= 0
                    return 0
                os.rmdir(os.path.join(conf['MODEL']['cur_loss_savepath'], f"{ru_id}.pkl"))
        else:
            torch.multiprocessing.spawn(trainer, nprocs =n_gpu, args=(conf, train_df, ru_id,logger))
            logger.info(f'{cur_state_str} remain[{len(ru_ids)}] training done')
            return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KT Anomaly Detection')
    parser.add_argument('--config', type=str)
    parser.add_argument('--loglevel', type=str, default='debug', choices=['debug', 'info'])
    parser.add_argument('--ru_ids_path', type=str)

    args = parser.parse_args()

    with open(os.path.join('./configs', args.config)) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    conf['LOG']['loglevel'] = args.loglevel

    fh = logging.FileHandler(conf['LOG']['logpath'])
    fh.setLevel(logging.DEBUG if conf['LOG']['loglevel'] == 'debug' else logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG if conf['LOG']['loglevel'] == 'debug' else logging.INFO)

    logger.info(args)
    logger.info(conf)

    if args.ru_ids_path == '-1':
        conn_manager = ConnManager(conf['DB'])
        ru_ids = get_ru_unique_list(conn_manager.conn)
        conn_manager.conn.close()
    else:
        # ru_df = pd.read_csv(args.ru_ids_path)
        # ru_ids = ru_df['RU_ID'].value_counts().index.tolist()
        ru_df, default_list, revision_list = read_target_csv(args.ru_ids_path)

    logger.info(f'The number of unique ru_ids is {len(ru_df)}(default:{len(default_list)}, revision:{len(revision_list)})')

    revision_done = False
    conditioned_state= 0
    while len(default_list) > 0:
        while len(revision_list) > 0:
            ru_id = revision_list.pop()
            duration = ru_df.loc[ru_id, 'DURATION']
            duration = duration if type(duration) is str else None
            # training
            train_ru(conf, ru_id, revision_list, duration, isrevision=True)
            if ru_id in default_list:
                default_list.remove(ru_id)

            revision_done = True

        if revision_done:
            # write revision as 0
            ru_df.loc[ru_df.REVISION_FLAG == 1, 'REVISION_FLAG'] = 0
            ru_df.to_csv(args.ru_ids_path)
            revision_done = False

        ru_id = default_list.pop()
        duration = ru_df.loc[ru_id, 'DURATION']
        duration = duration if type(duration) is str else None

        # training
        tmp_conditioned_state =train_ru(conf, ru_id, default_list, duration, isrevision=False)
        conditioned_state= conditioned_state+tmp_conditioned_state

        # update revision list
        ru_df, _, revision_list = read_target_csv(args.ru_ids_path)
    print("================================")
    print("The number of Conditioned RUs :", conditioned_state)
    print("================================")
