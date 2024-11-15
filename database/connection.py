from sqlalchemy import create_engine
import psycopg2
import io
import pandas as pd
import datetime
import glob
import os
import argparse


class ConnManager:
    def __init__(self, args):
        self.conn_string = f"host='{args['host']}' " \
                      f"port='{args['port']}' " \
                      f"dbname='{args['dbname']}' " \
                      f"user='{args['user']}' " \
                      f"password='{args['password']}'"

        self.conn = psycopg2.connect(self.conn_string)

    def update_conn(self):
        self.conn = psycopg2.connect(self.conn_string)

    def execute_select(self, sql):
        results = None
        with self.conn.cursor() as cur:
            cur.execute(sql)
            results = cur.fetchall()
            desc = cur.description
        return results, desc

    def execute_update(self, sql):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--host', type=str)
    parser.add_argument('--port', type=str)
    parser.add_argument('--dbname', type=str)
    parser.add_argument('--user', type=str)
    parser.add_argument('--password', type=str)

    args = parser.parse_args()

    dbconf = vars(args)
    conn_manager = ConnManager(dbconf)
    #sql = """SELECT * FROM ktsample ORDER BY datetime DESC"""  # debug
    sql = """SELECT * FROM tr_stat_5g_cu_du_5min"""
    results, desc = conn_manager.execute_select(sql)
    df = pd.DataFrame(results)
    df.columns = [d[0] for d in desc]
    print(df.head())
    conn_manager.update_conn()
    print('update conn')
    results, desc = conn_manager.execute_select(sql)
    df = pd.DataFrame(results)
    df.columns = [d[0] for d in desc]
    print(df.head())
    print('test done')





