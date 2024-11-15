import sys
#sys.path.append('..')
import psycopg2
import io
import pandas as pd
import datetime
import glob
import os
import argparse
import logging

from database.connection import ConnManager
from database.logger import logger, timed, formatter


@timed
def get_lasttime_from_db(conn):
    last_datetime = None
    with conn.cursor() as cur:
        cur.execute("""SELECT datetime FROM tbl_stat_5g_cu_du_5min_cur WHERE datetime IS NOT NULL ORDER BY datetime DESC LIMIT 1""")
        last_datetime = cur.fetchall()[0][0]
    return last_datetime


@timed
def get_lastdf_from_db(conn, start_time, end_time):
    last_df = None
    with conn.cursor() as cur:
        # kst_now = kst_now.strftime("%Y-%m-%d %H:%M:%S")
        #cur.execute(f"""SELECT * FROM tbl_stat_5g_cu_du_5min_cur WHERE datetime >= '{start_time}' and datetime <= '{end_time}' ORDER BY datetime DESC""")
        cur.execute(f"""SELECT * FROM tbl_stat_5g_cu_du_5min_cur WHERE datetime >= '{start_time}' and datetime <= '{end_time}' and flag IS DISTINCT FROM 1 ORDER BY datetime DESC""")
        last_df = pd.DataFrame(cur.fetchall())
        db_columns = [desc[0] for desc in cur.description]
        if len(last_df.columns) == len(db_columns):
            last_df.columns = [desc[0] for desc in cur.description]
            #last_df.set_index('index')
            print(last_df.columns)
    return last_df


def get_inference_data(args, logger):

    logger.debug(args)
    conn_manager = ConnManager(args)

    lastmin = args['lastmin']
    last_datetime = get_lasttime_from_db(conn_manager.conn)
    start_datetime = last_datetime - datetime.timedelta(minutes=lastmin)
    logger.debug(f"{start_datetime} ~ {last_datetime}")

    last_df = get_lastdf_from_db(conn_manager.conn,
                                 start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                                 last_datetime.strftime("%Y-%m-%d %H:%M:%S"))
    if len(last_df) == 0:
        logger.debug('Empty DataFrame')
    else:
        last_df.drop(['flag'], inplace=True, axis=1)
        logger.debug(f"DataFrame # rows:{len(last_df)}\n{last_df.head()}")

        if args['local_save']:
            savename = f"{last_datetime.strftime('%Y-%m-%d_%H:%M:%S')}.csv"
            savepath = os.path.join(args.savepath, savename)
            last_df.to_csv(savepath, index=False)
            logger.debug(f'inference data saved to {savepath}')

    logger.debug('TEST DONE')

    return last_df, start_datetime, last_datetime


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DB configuration')
    parser.add_argument('--host', type=str)
    parser.add_argument('--port', type=str)
    parser.add_argument('--dbname', type=str)
    parser.add_argument('--user', type=str)
    parser.add_argument('--password', type=str)
    parser.add_argument('--savepath', type=str, default='./data/temp/inference')
    parser.add_argument('--lastmin', type=int, default=5)

    args = parser.parse_args()
    fh = logging.FileHandler('./data/logs/inference_debug.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    get_inference_data(vars(args), logger)
