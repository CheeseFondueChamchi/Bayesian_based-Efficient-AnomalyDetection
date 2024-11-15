import os
import argparse
import yaml
import pandas as pd

import logging
from logging import handlers
from database.logger import logger, timed, formatter
from database.connection import ConnManager
from database.get_inference_data import get_inference_data
from models.evaluator import inference_and_insert
from database.get_train_data import get_ru_unique_list
from database.sqls import update_complete_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KT Anomaly Detection')
    parser.add_argument('--config', type=str)
    parser.add_argument('--ai-server-id', type=str)
    parser.add_argument('--loglevel', type=str, default='debug', choices=['debug', 'info'])
    parser.add_argument('--ru_ids_path', type=str, default='-1')

    args = parser.parse_args()

    with open(os.path.join('./configs', args.config)) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    conf['LOG']['loglevel'] = args.loglevel

    fh = logging.handlers.TimedRotatingFileHandler(filename = conf['LOG']['logpath'], when='midnight',interval=1)
    # fh = logging.FileHandler(conf['LOG']['logpath'])
    fh.setLevel(logging.DEBUG if conf['LOG']['loglevel'] == 'debug' else logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG if conf['LOG']['loglevel'] == 'debug' else logging.INFO)

    logger.info(args)
    logger.info(conf)

    conn_manager = ConnManager(conf['DB'])
    if 'kaist' in args.config:
        conn = conn_manager.conn
        cur = conn.cursor()
        sql = """UPDATE tbl_stat_5g_cu_du_5min_cur SET flag = NULL"""
        cur.execute(sql)
        conn.commit()
        logger.debug('initialize flag value in tbl_stat_5g_cu_du_5min_cur table')

    # get inference data from database
    X_data, start_time, last_time = get_inference_data(conf['DB'], logger)

    if args.ru_ids_path == '-1':
        ru_ids = get_ru_unique_list(conn_manager.conn)
    else:
        ru_df = pd.read_csv(args.ru_ids_path)
        ru_ids = ru_df['RU_ID'].value_counts().index.tolist()

    if len(X_data) != 0:
        # inference by models
        X_data = X_data.loc[X_data.ru_id.isin(ru_ids)]
        inference_and_insert(conf, logger, data=X_data)

        # update complete table
        conn = conn_manager.conn
        cur = conn.cursor()
        sql = update_complete_time(table_name=f'tbl_ai_output_{args.ai_server_id}', datetime=last_time)
        cur.execute(sql)
        print(sql)
        conn.commit()


