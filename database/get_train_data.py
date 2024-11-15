import sys
sys.path.append('..')
import pandas as pd
import datetime
import os
import argparse
import logging

from database.connection import ConnManager
from database.logger import logger, timed, formatter

@timed
def get_train_df_from_db(conn, du_id=None, cell_id=None):
    with conn.cursor() as cur:
        if du_id is None:
            cur.execute(f"""SELECT * FROM tbl_stat_5g_cu_du_5min ORDER BY datetime DESC""")
        else:
            cur.execute(f"""SELECT * FROM tbl_stat_5g_cu_du_5min WHERE du_id='{du_id}' AND cell_id={cell_id} ORDER BY datetime DESC""")
        train_df = pd.DataFrame(cur.fetchall())
        db_columns = [desc[0] for desc in cur.description]
        #         print(db_columns)
        if len(train_df.columns) == len(db_columns):
            train_df.columns = [desc[0] for desc in cur.description]
            #train_df.set_index('index')

    return train_df

@timed
def get_du_cu_unique_list(conn):
    with conn.cursor() as cur:
        cur.execute(f"""SELECT DISTINCT du_id, cell_id FROM tbl_stat_5g_cu_du_5min;""")
        du_cell_df = pd.DataFrame(cur.fetchall())
        db_columns = [desc[0] for desc in cur.description]
        if len(du_cell_df.columns) == len(db_columns):
            du_cell_df.columns = [desc[0] for desc in cur.description]

    return du_cell_df

@timed
def get_ru_unique_list(conn):
    with conn.cursor() as cur:
        cur.execute(f"""SELECT DISTINCT ru_id FROM tbl_stat_5g_cu_du_5min;""")
        du_cell_df = pd.DataFrame(cur.fetchall())
        db_columns = [desc[0] for desc in cur.description]
        if len(du_cell_df.columns) == len(db_columns):
            du_cell_df.columns = [desc[0] for desc in cur.description]

    return du_cell_df['ru_id'].tolist()


@timed
def get_train_data_by_ru(conn, ru_id, st=None, ed=None):
    if (st is None) and (ed is None):
        sql = f"""SELECT * FROM tbl_stat_5g_cu_du_5min WHERE ru_id='{ru_id}' ORDER BY datetime DESC limit 4032"""
    elif (st is not None) and (ed is None):
        sql = f"""SELECT * FROM tbl_stat_5g_cu_du_5min WHERE ru_id='{ru_id}' and datetime >= '{st}' ORDER BY datetime DESC"""
    elif (st is None) and (ed is not None):
        sql = f"""SELECT * FROM tbl_stat_5g_cu_du_5min WHERE ru_id='{ru_id}' and datetime <= '{ed}' ORDER BY datetime DESC"""
    else:
        sql = f"""SELECT * FROM tbl_stat_5g_cu_du_5min WHERE ru_id='{ru_id}' and datetime >= '{st}' and datetime <= '{ed}' ORDER BY datetime DESC"""
    with conn.cursor() as cur:
        cur.execute(sql)
        train_df = pd.DataFrame(cur.fetchall())
        db_columns = [desc[0] for desc in cur.description]
        if len(train_df.columns) == len(db_columns):
            train_df.columns = [desc[0] for desc in cur.description]
    return train_df


def get_xgboost_train_data(conn, st=None, ed=None):
    if (st is None) and (ed is None):
        sql = f"""SELECT * FROM tm_rca_data d INNER JOIN tm_rca r ON d.id = r.id ORDER BY datetime DESC"""
    elif (st is not None) and (ed is None):
        sql = f"""SELECT * FROM tm_rca_data d INNER JOIN tm_rca r ON d.id = r.id WHERE d.datetime >= '{st}' ORDER BY datetime DESC"""
    elif (st is None) and (ed is not None):
        sql = f"""SELECT * FROM tm_rca_data d INNER JOIN tm_rca r ON d.id = r.id WHERE d.datetime <= '{ed}' ORDER BY datetime DESC"""
    else:
        sql = f"""SELECT * FROM tm_rca_data d INNER JOIN tm_rca r ON d.id = r.id WHERE d.datetime >= '{st}' and d.datetime <= '{ed}' ORDER BY datetime DESC"""

    with conn.cursor() as cur:
        cur.execute(sql)
        train_df = pd.DataFrame(cur.fetchall())
        db_columns = [desc[0] for desc in cur.description]
        if len(train_df.columns) == len(db_columns):
            train_df.columns = [desc[0] for desc in cur.description]
    return train_df



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DB configuration')
    parser.add_argument('--host', type=str)
    parser.add_argument('--port', type=str)
    parser.add_argument('--dbname', type=str)
    parser.add_argument('--user', type=str)
    parser.add_argument('--password', type=str)
    parser.add_argument('--savepath', type=str, default='../data/temp/training')

    args = parser.parse_args()

    fh = logging.FileHandler('../data/logs/train_debug.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.debug(args)

    conn_manager = ConnManager(args)

    du_cell_df = get_du_cu_unique_list(conn_manager.conn)
    logger.debug(f"DataFrame # rows:{len(du_cell_df)}\n{du_cell_df.head()}")
    du_id, cell_id = du_cell_df.iloc[0]

    train_dfs = []
    for i, row in du_cell_df.iterrows():
        du_id = row['du_id']
        cell_id = row['cell_id']
        print(i, du_id, cell_id)
        train_df = get_train_df_from_db(conn_manager.conn, du_id=du_id, cell_id=cell_id)
        train_dfs.append(train_df)
        if i == 4:
            break

    train_df = pd.concat(train_dfs, axis=0)

    if len(train_df) == 0:
        logger.debug('Empty DataFrame')
    else:
        # last_df.info()
        #train_df.drop(['id', 'flag'], inplace=True, axis=1)
        train_df.drop(['flag'], inplace=True, axis=1)
        logger.debug(f"DataFrame # rows:{len(train_df)}\n{train_df.head()}")
        now = datetime.datetime.now()
        savename = f"{now.strftime('%Y-%m-%d_%H:%M:%S')}.csv"
        savepath = os.path.join(args.savepath, savename)
        train_df.to_csv(savepath, index=False)

    logger.debug('TEST DONE')
