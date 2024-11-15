import sys

sys.path.append('..')
import numpy as np
import pandas as pd


def calc_z_score(cell_data, te):
    # feature_high = np.array(['ErabAddSucc',  'ErabAddRate', 'EndcAddSucc', 'EndcAddRate',
    #                          'EndcModByMenbSucc', 'EndcModByMenbRate', 'EndcModBySgnbSucc', 'EndcModBySgnbRate',
    #                          'ConnEstabRate', 'EndcChgSucc', 'EndcChgRate', 'HandoverSucc',
    #                          'HandoverRate', 'ReEstabSucc', 'ReEstabRate', 'DLReceivedRiAvg',
    #                          'DLTransmittedMcsAvg', 'NumRAR', 'NumMSG3', 'RachRARRate'])
    # print(te)
    feature_high = np.array(['uenoavg', 'uenomax', 'erabaddsucc', 'erabaddrate', 'endcaddsucc', 'endcaddrate', 'endcmodbymenbsucc',
                             'endcmodbymenbrate', 'endcmodbysgnbsucc', 'endcmodbysgnbrate', 'endcchgsucc',
                             'endcchgrate', 'dlreceivedriavg', 'dltransmittedmcsavg', 'numrar',
                             'nummsg3', 'rachrarrate', 'powerheadroomreport', 'dlreceivedcqiavg'])
    # 210924 z_score update
    # feature_high_cnt = np.array(['erabaddsucc', 'endcaddsucc', 'endcmodbymenbsucc', 'endcmodbysgnbsucc', 'endcchgsucc', 'numrar', 'nummsg3'])
    feature_high_cnt = ['erabaddsucc', 'endcaddsucc', 'endcmodbymenbsucc', 'endcmodbysgnbsucc', 'endcchgsucc', 'numrar', 'nummsg3']

    # feature_mid = np.array(['UeNoAvg', 'UeNoMax', 'ErabAddAtt', 'EndcAddAtt', 'EndcModByMenbAtt', 'EndcModBySgnbAtt',
    #                         'ConnEstabAtt', 'ConnEstabSucc', 'EndcChgAtt', 'HandoverAtt', 'ReEstabAtt', 'RlcULByte',
    #                         'RlcDLByte', 'AirMacULByte', 'AirMacDLByte', 'RachPreambleA', 'AttPaging'])
    # TODO: 0803
    feature_mid = np.array(['erabaddatt', 'endcaddatt', 'endcmodbymenbatt', 'endcmodbysgnbatt',
                            'endcchgatt', 'rlculbyte', 'rlcdlbyte', 'airmaculbyte', 'airmacdlbyte', 'rachpreamblea'])
    
    # 210924 z_score test
    feature_mid_cnt = np.array(['erabaddatt', 'endcaddatt', 'endcmodbymenbatt', 'endcmodbysgnbatt',
                            'endcchgatt', 'rlculbyte', 'rlcdlbyte', 'airmaculbyte', 'airmacdlbyte', 'rachpreamblea'])

    # feature_low = np.array(['SCGFail', 'SCGFailRatio', 'CSLCuCp', 'RedirectionToLTE_CoverageOut',
    #                         'RedirectionToLTE_EPSFallback', 'RedirectionToLTE_EmergencyFallback', 'ReEstabRatio',
    #                         'TotPrbULAvg', 'TotPrbDLAvg', 'CSLDu ', 'BLER_UL', 'BLER_DL', 'RssiPathAvg'])
    feature_low = np.array(['scgfail', 'scgfailratio', 'cslcucp', 'totprbulavg', 'totprbdlavg', 'csldu', 'bler_ul',
                            'bler_dl', 'rssipathavg', 'outofsynccount','endcrelbymenb','scgfailoper_dl_t310expiry_count','scgfailoper_ul_rlcmaxnumretx_count','scgfailoper_dl_rlcmaxnumretx_count'])
    feature_low_dis = np.array(['scgfail', 'cslcucp', 'csldu', 'outofsynccount','endcrelbymenb','scgfailoper_dl_t310expiry_count','scgfailoper_ul_rlcmaxnumretx_count','scgfailoper_dl_rlcmaxnumretx_count'])
    feature_low_cnt = np.array(['totprbulavg', 'totprbdlavg'])

    cls_feature_name = np.array(['scgfail', 'totprbdlavg', 'erabaddrate', 'uenomax', 'endcmodbymenbrate', 'cslcucp',
                   'endcmodbysgnbrate', 'endcchgsucc', 'erabaddatt', 'endcchgatt', 'numrar', 'endcmodbysgnbatt',
                   'rlcdlbyte', 'airmaculbyte', 'uenoavg', 'endcchgrate', 'airmacdlbyte', 'dltransmittedmcsavg',
                   'bler_ul', 'endcmodbymenbsucc', 'endcaddsucc', 'rssipathavg', 'endcaddrate', 'erabaddsucc',
                   'rlculbyte', 'totprbulavg', 'csldu', 'bler_dl', 'endcaddatt', 'endcmodbysgnbsucc', 'outofsynccount',
                   'dlreceivedcqiavg', 'endcmodbymenbatt', 'nummsg3', 'powerheadroomreport', 'scgfailratio',
                   'dlreceivedriavg', 'rachrarrate', 'rachpreamblea','endcrelbymenb','scgfailoper_dl_t310expiry_count','scgfailoper_ul_rlcmaxnumretx_count','scgfailoper_dl_rlcmaxnumretx_count'])

    data_z_score = np.zeros(cell_data.shape[0])
    v_RachPA = cell_data['rachpreamblea']
    # 210924 z_score test
    data_z_score_scale = np.zeros(cell_data.shape[0])

    mean_all = te['feature_mean']
    std_all = te['feature_std']

    # Feature high
    weight_high = np.ones(len(feature_high))

    mean = mean_all[feature_high].to_numpy().reshape(1, -1)
    std = std_all[feature_high].to_numpy().reshape(1, -1)

    z_score_1 = (cell_data[feature_high] - mean) / std
    z_score_1 = z_score_1.replace([np.inf, -np.inf], np.nan).fillna(0.001)
    z_score_1[z_score_1 >= 0] = 0
    z_score_1[z_score_1 < 0] = z_score_1.abs()
    z_score_1 = z_score_1 * weight_high

    z_score_1_scale = z_score_1.copy().astype(float)
    temp = z_score_1[feature_high_cnt].copy().astype(float)
    z_score_1[feature_high_cnt] = 1.7 * np.log10(1 + temp) / np.log10(3) + 0.15 * temp
    z_score_1_scale[feature_high_cnt] = 1.8 * np.log10(1 + temp) / np.log10(3) + 0.1 * temp

    z_score = z_score_1.sum(1)
    data_z_score += z_score

    z_score_scale = z_score_1_scale.sum(1)
    data_z_score_scale += z_score_scale

    z_score_1_norm = z_score_1**2

    # Feature mid
    weight_mid = np.ones(len(feature_mid))

    mean = mean_all.loc[feature_mid].to_numpy().reshape(1, -1)
    std = std_all.loc[feature_mid].to_numpy().reshape(1, -1)

    z_score_2 = (cell_data[feature_mid] - mean) / std
    z_score_2 = z_score_2.replace([np.inf, -np.inf], np.nan).fillna(0.001)
    z_score_2 = z_score_2.abs()
    z_score_2 = z_score_2 * weight_mid

    z_score_2_scale = z_score_2.copy().astype(float)
    temp = z_score_2[feature_mid_cnt].copy().astype(float)
    z_score_2[feature_mid_cnt] = 1.7 * np.log10(1 + temp) / np.log10(3) + 0.15 * temp
    z_score_2_scale[feature_mid_cnt] = 1.8 * np.log10(1 + temp) / np.log10(3) + 0.1 * temp

    z_score = z_score_2.sum(1)
    data_z_score += z_score

    z_score_scale = z_score_2_scale.sum(1)
    data_z_score_scale += z_score_scale

    z_score_2_norm = z_score_2**2

    # Feature low
    weight_low = np.ones(len(feature_low))

    mean = mean_all.loc[feature_low].to_numpy().reshape(1, -1)
    std = std_all.loc[feature_low].to_numpy().reshape(1, -1)

    z_score_3 = (cell_data[feature_low] - mean) / std
    z_score_3 = z_score_3.replace([np.inf, -np.inf], np.nan).fillna(-0.001)
    z_score_3[z_score_3 <= 0] = 0
    z_score_3[z_score_3 > 0] = z_score_3.abs()
    z_score_3 = z_score_3 * weight_low

    z_score_3_scale = z_score_3.copy().astype(float)

    temp = z_score_3[feature_low_dis].copy().astype(float)
    z_score_3[feature_low_dis] = 1.5 * np.log10(1 + temp) / np.log10(3) + 0.25 * temp
    z_score_3_scale[feature_low_dis] = 1.6 * np.log10(1 + temp) / np.log10(3) + 0.2 * temp

    temp = z_score_3[feature_low_cnt].copy().astype(float)
    z_score_3[feature_low_cnt] = 1.7 * np.log10(1 + temp) / np.log10(3) + 0.15 * temp
    z_score_3_scale[feature_low_cnt] = 1.8 * np.log10(1 + temp) / np.log10(3) + 0.1 * temp

    z_score = z_score_3.sum(1)
    data_z_score += z_score

    z_score_scale = z_score_3_scale.sum(1)
    data_z_score_scale += z_score_scale

    z_score_3_norm = z_score_3**2
    z_score_norm = 2*(z_score_1_norm.sum(1) + z_score_2_norm.sum(1) + z_score_3_norm.sum(1))**0.5
    

        # TODO: Check 0802
    # z_score_all = pd.concat([z_score_1, z_score_2, z_score_3], 1)
    # feature_name_all = np.concatenate([feature_high, feature_mid, feature_low])
    feature_name_all = np.concatenate([feature_high, feature_mid, feature_low])

    mean = mean_all[feature_name_all].to_numpy().reshape(1, -1)
    std = std_all[feature_name_all].to_numpy().reshape(1, -1)

    z_score_all = (cell_data[feature_name_all] - mean) / std
    z_score_all = z_score_all.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    cls_feature = z_score_all[cls_feature_name]


    # top k
    top_5_feature = []
    top_5_val = []
    total_feature = []
    z_score = z_score_all.to_numpy()

    for i in range(len(z_score)):
        temp = z_score[i]
        temp_abs = np.abs(temp)
        temp_5_idx = temp_abs.argsort()[-10:][::-1]
        temp_5_feature = feature_name_all[temp_5_idx]
        temp_5_val = temp[temp_5_idx]
        top_5_feature.append(temp_5_feature)
        top_5_val.append(temp_5_val)
        total_feature.append(feature_name_all)

    top_5_feature = np.array(top_5_feature)
    top_5_val = np.array(top_5_val)
    total_feature = np.array(total_feature)
    
    # 210924 z_score test
    return data_z_score, top_5_feature, top_5_val, total_feature, z_score_all, cls_feature, data_z_score_scale, z_score_norm


def calc_z_score_backup(cell_data, te):
    # print(cell_data.columns)

    feature_mid = cell_data.columns
    feature_mid = np.array(feature_mid)
    # feature_mid = np.array(['UeNoAvg', 'UeNoMax', 'SCGFail', 'ErabAddAtt', 'ErabAddSucc', 'EndcAddAtt', 'EndcAddSucc',
    #                'EndcModByMenbAtt', 'EndcModByMenbSucc', 'EndcModBySgnbAtt', 'EndcModBySgnbSucc', 'CSLCuCp',
    #                'RrcEstabAtt', 'RrcEstabSucc', 'CoverageOut', 'EPSFallback', 'EmergencyFallback', 'EndcIntraChgAtt',
    #                'EndcIntraChgSucc', 'EndcInterChgSrcAtt', 'EndcInterChgSrcSucc', 'HandoverAtt', 'HandoverSucc',
    #                'ReEstabAtt', 'ReEstabSucc', 'RlcULByte', 'RlcDLByte', 'PrbUL', 'PrbDL', 'RIAvg', 'MCSDL', 'CSLDu',
    #                'AirMacULByte', 'AirMacDLByte', 'BLERof1stTxTrial_UL', 'BLERofxthTxTrial_UL', 'BLERof1stTxTrial_DL',
    #                'BLERofxthTxTrial_DL', 'RachAtt', 'RachRAR', 'RachMSG3', 'DeniedmoData', 'AttPaging', 'RssiPathAvg',
    #                'ErabAddRate', 'EndcAddRate', 'EndcModByMenbRate', 'EndcModBySgnbRate', 'RrcEstabRate',
    #                'EndcIntraChgRate', 'EndcInterChgRate', 'HandoverRate', 'SCGFailRatio', 'ReEstabRate',
    #                'ReEstabRatio', 'BLER_UL', 'BLER_DL', 'RachRARRate', 'PagingOverload'])

    weight_mid = np.ones(len(feature_mid))

    data_z_score = np.zeros(cell_data.shape[0])
    mean_all = te['feature_mean']
    std_all = te['feature_std']

    mean = mean_all.loc[feature_mid].to_numpy().reshape(1, -1)
    std = std_all.loc[feature_mid].to_numpy().reshape(1, -1)

    z_score = (cell_data[feature_mid] - mean) / std
    z_score = z_score.replace([np.inf, -np.inf], np.nan).fillna(0.1)
    z_score = z_score.abs()
    z_score = z_score * weight_mid

    # sum
    data_z_score += z_score.sum(1)
    data_z_score = data_z_score.to_numpy()

    # top k
    top_5_feature = []
    top_5_val = []
    total_feature = []
    z_score = z_score.to_numpy()
    for i in range(len(z_score)):
        temp = z_score[i]
        temp_5_idx = temp.argsort()[-5:][::-1]
        temp_5_feature = feature_mid[temp_5_idx]
        temp_5_val = temp[temp_5_idx]
        top_5_feature.append(temp_5_feature)
        top_5_val.append(temp_5_val)
        total_feature.append(feature_mid)

    top_5_feature = np.array(top_5_feature)
    top_5_val = np.array(top_5_val)
    total_feature = np.array(total_feature)

    return data_z_score, top_5_feature, top_5_val, total_feature, z_score
