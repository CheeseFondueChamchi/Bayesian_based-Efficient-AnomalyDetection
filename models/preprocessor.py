

import pandas as pd
import numpy as np
import os
import pickle
from database.logger import logger, timed, formatter

not_value_features = ['datetime', 'du_id', 'cell_id', 'ru_id', 'gnb_id', 'acpf_id', 'flag']
not_use_features = ['connestabatt', 'connestabsucc', 'connestabrate', 'redirectiontolte_coverageout',
                    'redirectiontolte_epsfallback', 'redirectiontolte_emergencyfallback', 'handoveratt',
                    'handoversucc', 'handoverrate', 'reestabatt', 'reestabsucc', 'reestabrate', 'reestabratio',
                    'attpaging']
                    
log_features_small_value = ['rlculbyte', 'rlcdlbyte', 'airmaculbyte', 'airmacdlbyte']
log_features_large_value = ['rachpreamblea', 'numrar']


def preprocess_features(args, raw_df: pd.DataFrame):
    rus = raw_df['ru_id'].value_counts().index.tolist()
    if len(rus) != 1:
        raise Exception(f'Expect only one ru_id, but there are {len(rus)}')
    ru = str(rus[0])

    rm_features = raw_df.columns[raw_df.columns.isin(not_value_features)]
    temp_df = raw_df.drop(rm_features, axis=1).copy()
    temp_df = temp_df.astype(float)
    temp_df = temp_df[temp_df['endcaddatt'] != 0]

    # temp_df = making_new_features(temp_df)  # 57
    temp_df = temp_df.dropna(0)

    if len(temp_df) == 0:
        logger.debug(f'There is no data passed in {ru}')
        return None

    temp_df = temp_df.drop(not_use_features, axis=1)
    temp_df[log_features_small_value + log_features_large_value] = np.log10(temp_df[log_features_small_value + log_features_large_value])
    temp_df = temp_df.replace([np.inf, -np.inf], np.nan)
    temp_df[log_features_small_value] = temp_df[log_features_small_value].fillna(-70)
    temp_df[log_features_large_value] = temp_df[log_features_large_value].fillna(-70)

    return temp_df


def normalize_features(args, ru: str, df: pd.DataFrame, istrain: bool):

    if istrain:
        feature_mean = df.mean()
        feature_std = df.std()

        std_list_1 = ['scgfail', 'scgfailratio', 'erabaddrate', 'endcaddrate',
                      'endcmodbymenbatt', 'endcmodbymenbsucc', 'endcmodbymenbrate', 'endcmodbysgnbrate',
                      'endcchgrate','endcmodbysgnbatt','endcchgatt','endcrelbymenb']
        std_list_2 = ['rachrarrate']
        std_list_3 = ['outofsynccount','totprbulavg','totprbdlavg','scgfailoper_dl_rlcmaxnumretx_count','scgfailoper_ul_rlcmaxnumretx_count']

        temp = feature_std.loc[std_list_1].copy()
        temp[temp < 1] = 1
        feature_std.loc[std_list_1] = temp

        temp = feature_std.loc[std_list_2].copy()
        temp[temp < 2] = 2
        feature_std.loc[std_list_2] = temp

        temp = feature_std.loc[std_list_3].copy()
        temp[temp < 0.5] = 0.5
        feature_std.loc[std_list_3] = temp
        statistics_dict = {'feature_mean': feature_mean, 'feature_std': feature_std, 'col_name': df.columns}
        with open(os.path.join(args['train_stat_savepath'], f'{ru}.pickle'), "wb") as f:
            pickle.dump(statistics_dict, f)
    else:
        with open(os.path.join(args['train_stat_savepath'], f'{ru}.pickle'), "rb") as f:
            statistics_dict = pickle.load(f)
        feature_mean = statistics_dict['feature_mean'].values
        feature_std = statistics_dict['feature_std'].values
    
    df = (df - feature_mean) / (feature_std + 1e-6)
    
    return df, statistics_dict


def making_new_features(raw_df: pd.DataFrame):
    '''
    Rule-based preprocessing for making new 15 feature columns

    raw_df: pd.DataFrame (data size x raw data dim)

    return processed_df
    '''

    processed_df = raw_df.copy()
    print(processed_df.columns)

    # 1. ErabAddRate
    try:
        processed_df.loc[:, 'erabauddrate'] = (processed_df['erabaddsucc'] / processed_df['erabaddatt']) * 100
        processed_df.loc[processed_df['erabaddatt'] == 0, 'erabaddrate'] = 0
    except Exception as e:
        logger.debug(f'{e.__class__.__name__}: {e} in processing ErabAddRate')
        processed_df.loc[:, 'erabaddrate'] = np.zeros(raw_df.shape[0])

    # 2. EndcAddRate
    try:
        processed_df.loc[:, 'endcaddrate'] = (processed_df['endcaddsucc'] / processed_df['endcaddatt']) * 100
        processed_df.loc[processed_df['andcaddatt'] == 0, 'endcaddrate'] = 0
    except Exception as e:
        logger.debug(f'{e.__class__.__name__}: {e} in processing EndcAddRate')
        processed_df.loc[:, 'endcaddrate'] = np.zeros(raw_df.shape[0])

    # 3. EndcModByMenbRate
    try:
        processed_df.loc[:,'endcmodbymenbrate'] = (processed_df['endcmodbymenbsucc'] / processed_df['endcmodbymenbatt']) * 100
        processed_df.loc[processed_df['endcmodbymenbatt'] == 0, 'endcmodbymenbrate'] = 0
    except Exception as e:
        logger.debug(f'{e.__class__.__name__}: {e} in processing EndcModByMenbRate')
        processed_df.loc[:, 'endcmodbymenbrate'] = np.zeros(raw_df.shape[0])

    # 4. EndcModBySgnbRate
    try:
        processed_df.loc[:,'endcmodbysgnbrate'] = (processed_df['endcmodbysgnbsucc'] / processed_df['endcmodbysgnbatt']) * 100
        processed_df.loc[processed_df['endcmodbysgnbatt'] == 0, 'endcmodbysgnbrate'] = 0
    except Exception as e:
        logger.debug(f'{e.__class__.__name__}: {e} in processing EndcModBySgnbRate')
        processed_df.loc[:, 'endcmodbysgnbrate'] = np.zeros(raw_df.shape[0])

    # 5. RrcEstabRate
    try:
        processed_df.loc[:, 'rrcestabrate'] = (processed_df['rrcestabsucc'] / processed_df['rrcestabatt']) * 100
        processed_df.loc[processed_df['rrcestabatt'] == 0, 'rrcestabrate'] = 0
    except Exception as e:
        logger.debug(f'{e.__class__.__name__}: {e} in processing RrcEstabRate')
        processed_df.loc[:, 'rrcestabrate'] = np.zeros(raw_df.shape[0])

    # 6. EndcIntraChgRate
    try:
        processed_df.loc[:, 'endcintrachgrate'] = (processed_df['endcintrachgsucc'] / processed_df['endcintrachgatt']) * 100
        processed_df.loc[processed_df['endcintrachgatt'] == 0, 'endcintrachgrate'] = 0
    except Exception as e:
        logger.debug(f'{e.__class__.__name__}: {e} in processing EndcIntraChgRate')
        processed_df.loc[:, 'endcintrachgrate'] = np.zeros(raw_df.shape[0])

    # 7. EndcInterChgRate
    try:
        processed_df.loc[:, 'endcinterchgrate'] = (processed_df['endcinterchgsucc'] / processed_df['endcinterchgatt']) * 100
        processed_df.loc[processed_df['endcinterchgatt'] == 0, 'endcinterchgrate'] = 0
    except Exception as e:
        logger.debug(f'{e.__class__.__name__}: {e} in processing EndcInterChgRate')
        processed_df.loc[:, 'endcinterchgrate'] = np.zeros(raw_df.shape[0])

    # 8. HandoverRate
    try:
        processed_df.loc[:, 'handoverrate'] = (processed_df['handoversucc'] / processed_df['handoveratt']) * 100
        processed_df.loc[processed_df['handoveratt'] == 0, 'handoverrate'] = 0
    except Exception as e:
        logger.debug(f'{e.__class__.__name__}: {e} in processing HandoverRate')
        processed_df.loc[:, 'handoverrate'] = np.zeros(raw_df.shape[0])

    # 9. SCGFailRatio
    try:
        processed_df.loc[:, 'scgfailratio'] = (processed_df['scgfail'] / processed_df['erabaddsucc']) * 100
        processed_df.loc[processed_df['erabaddsucc'] == 0, 'scgfailratio'] = 0
    except Exception as e:
        logger.debug(f'{e.__class__.__name__}: {e} in processing SCGFailRatio')
        processed_df.loc[:, 'scgfailratio'] = np.zeros(raw_df.shape[0])

    # 10. ReEstabRate
    try:
        processed_df.loc[:, 'reestabrate'] = (processed_df['reestabsucc'] / processed_df['reestabatt']) * 100
        processed_df.loc[processed_df['reestabatt'] == 0, 'reestabrate'] = 0
    except Exception as e:
        logger.debug(f'{e.__class__.__name__}: {e} in processing ReEstabRate')
        processed_df.loc[:, 'reestabrate'] = np.zeros(raw_df.shape[0])

    # 11. ReEstabRatio
    try:
        processed_df.loc[:, 'reestabratio'] = (processed_df['reestabatt'] / processed_df['rrcestabsucc']) * 100
        processed_df.loc[processed_df['rrcestabsucc'] == 0, 'reestabratio'] = 0
    except Exception as e:
        logger.debug(f'{e.__class__.__name__}: {e} in processing ReEstabRatio')
        processed_df.loc[:, 'reestabratio'] = np.zeros(raw_df.shape[0])

    # 12. BLER_UL
    try:
        processed_df.loc[:, 'bler_ul'] = (processed_df['blerofxthtxtrial_ul'] / processed_df['blerof1sttxtrial_ul']) * 100
        processed_df.loc[processed_df['blerof1sttxtrial_ul'] == 0, 'bler_ul'] = 0
    except Exception as e:
        logger.debug(f'{e.__class__.__name__}: {e} in processing BLER_UL')
        processed_df.loc[:, 'bler_ul'] = np.zeros(raw_df.shape[0])

    # 13. BLER_DL
    try:
        processed_df.loc[:, 'bler_dl'] = (processed_df['blerofxthtxtrial_dl'] / processed_df['blerof1sttxtrial_dl']) * 100
        processed_df.loc[processed_df['blerof1sttxtrial_dl'] == 0, 'bler_dl'] = 0
    except Exception as e:
        logger.debug(f'{e.__class__.__name__}: {e} in processing BLER_DL')
        processed_df.loc[:, 'bler_dl'] = np.zeros(raw_df.shape[0])

    # 14. RachRARRate
    try:
        processed_df.loc[:, 'rachrarrate'] = (processed_df['rachrar'] / processed_df['rachatt']) * 100
        processed_df.loc[processed_df['rachatt'] == 0, 'rachrarrate'] = 0
    except Exception as e:
        logger.debug(f'{e.__class__.__name__}: {e} in processing RachRARRate')
        processed_df.loc[:, 'rachrarrate'] = np.zeros(raw_df.shape[0])

    # 15. PagingOverload
    try:
        C = 1
        processed_df.loc[:, 'pagingoverload'] = (processed_df['attpaging'] / C) * 100
    except Exception as e:
        logger.debug(f'{e.__class__.__name__}: {e} in processing PagingOverload')
        processed_df.loc[:, 'pagingoverload'] = np.zeros(raw_df.shape[0])

    return processed_df


if __name__ == '__main__':
    data_path = '../data/temp/training/train_sample.csv'
    processed_df = making_new_features(pd.read_csv(data_path))

    save_path = '../data/temp/training/processed_df.csv'
    processed_df.to_csv(save_path)
