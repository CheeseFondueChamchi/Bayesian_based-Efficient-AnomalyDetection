import datetime as dt

def get_insert_inference_output_sql(**kwargs):
    datetime = kwargs['datetime']
    ru_id = kwargs['ru_id']
    cell_id = kwargs['cell_id']
    du_id = kwargs['du_id']
    score0 = kwargs['score0']
    score1 = 'NULL' if kwargs['score1'] is None else f"'{kwargs['score1'] }'"
    z_sum = kwargs['z_sum']
    f_list = kwargs['f_list']
    z_list = kwargs['z_list']
    f_tot = kwargs['f_tot']
    z_tot = kwargs['z_tot']
    rca_num = 'NULL' if kwargs['rca_num'] is None else f"'{kwargs['rca_num']}'"
    rca_str = 'NULL' if kwargs['rca_str'] is None else f"'{kwargs['rca_str']}'"
    act_num = 'NULL' if kwargs['act_num'] is None else f"'{kwargs['act_num']}'"
    act_str = 'NULL' if kwargs['act_str'] is None else f"'{kwargs['act_str']}'"
    spare0 = kwargs['spare0']
    spare1 = kwargs['spare1']
    spare2 = kwargs['spare2']
    # 210928 z_score test: test value(spare0) add
    inference_output_insert_sql = f"""INSERT INTO tbl_ai_output
                                            (
                                                datetime,
                                                ru_id, 
                                                cell_id,
                                                du_id,
                                                score0,
                                                score1,
                                                z_sum,
                                                f_list,
                                                z_list,
                                                f_tot,
                                                z_tot,
                                                rca_num,
                                                rca_str,
                                                act_num,
                                                act_str,
                                                spare0,
                                                spare1,
                                                spare2
                                            )
                                            VALUES
                                            (                       
                                                '{datetime}',
                                                '{ru_id}',
                                                '{cell_id}',
                                                '{du_id}',
                                                {score0},
                                                {score1},
                                                {z_sum},
                                                '{f_list}',
                                                '{z_list}',
                                                '{f_tot}',
                                                '{z_tot}',
                                                {rca_num},
                                                {rca_str},
                                                {act_num},
                                                {act_str},
                                                {spare0},
                                                {spare1},
                                                {spare2}
                                            );"""
    print(inference_output_insert_sql)
    return inference_output_insert_sql


def get_update_flag_sql(**kwargs):
    datetime = kwargs['datetime']
    ru_id = kwargs['ru_id']
    update_flag_sql = f"""UPDATE tbl_stat_5g_cu_du_5min_cur 
                            SET flag = 1 
                            WHERE datetime = '{datetime}' 
                            AND ru_id='{ru_id}';"""
    return update_flag_sql


def update_complete_time(**kwargs):
    #KST = dt.timezone(dt.timedelta(hours=9))
    #dt.datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')
    insert_datetime = kwargs['datetime']
    compl_table_name = kwargs['table_name']

    #insert_compl_sql = f""" insert into tbl_complete (table_name, datetime)
    #                            values ('{compl_table_name}', '{insert_datetime}')
     #                           on conflict (table_name)
      #                          do update set datetime = '{insert_datetime}'
       #                         where tbl_complete.datetime < '{insert_datetime}';                              
    

    update_compl_sql = f"""UPDATE tbl_complete
                             SET datetime = '{insert_datetime}'
                             WHERE table_name = '{compl_table_name}';
                             INSERT INTO tbl_complete_test
                             (table_name, datetime, input_time, cnt_flag)
                             values
                             ('{compl_table_name}','{insert_datetime}',current_timestamp(0),(select count(*) FROM tbl_stat_5g_cu_du_5min_cur where datetime = '{insert_datetime}' and flag = 1));"""

    return update_compl_sql
