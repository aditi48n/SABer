#!/usr/bin/env python

import logging
import multiprocessing
import sys
from functools import reduce

import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

pd.options.mode.chained_assignment = None  # default='warn'


def EArecruit(p):  # Error Analysis for all recruits per sag
    temp_id, temp_mh_df, temp_nmf_df, temp_all_df, temp_contig_df, \
    temp_src2contig_list, temp_src2strain_list = p
    temp_mh_df['minhash'] = 1
    temp_nmf_df['NMF'] = 1
    temp_all_df['All_Hz'] = 1
    temp_contig_df['sag_id'] = temp_id
    df_list = [temp_contig_df, temp_mh_df, temp_nmf_df, temp_all_df]
    merge_recruits_df = reduce(lambda left, right: pd.merge(left, right,
                                                            on=['sag_id', 'contig_id'],
                                                            how='left'),
                               df_list
                               )
    merge_recruits_df.fillna(-1, inplace=True)
    merge_recruits_df['exact_truth'] = [1 if x in temp_src2contig_list else -1
                                        for x in merge_recruits_df['contig_id']
                                        ]
    merge_recruits_df['strain_truth'] = [1 if x in temp_src2strain_list else -1
                                         for x in merge_recruits_df['contig_id']
                                         ]
    contig_id_list = list(merge_recruits_df['contig_id'])
    contig_bp_list = list(merge_recruits_df['bp_cnt'])
    exact_truth = list(merge_recruits_df['exact_truth'])
    strain_truth = list(merge_recruits_df['strain_truth'])
    algo_list = ['minhash', 'NMF', 'All_Hz']
    stats_lists = []
    for algo in algo_list:
        pred = list(merge_recruits_df[algo])
        stats_lists.extend(recruit_stats([temp_id, algo, contig_id_list, contig_bp_list,
                                          exact_truth, strain_truth, pred
                                          ]))
    return stats_lists


def recruit_stats(p):
    sag_id, algo, contig_id_list, contig_bp_list, exact_truth, strain_truth, pred = p
    pred_df = pd.DataFrame(zip(contig_id_list, contig_bp_list, pred),
                           columns=['contig_id', 'contig_bp', 'pred']
                           )
    pred_df['sag_id'] = sag_id
    pred_df['algorithm'] = algo
    pred_df = pred_df[['sag_id', 'algorithm', 'contig_id', 'contig_bp', 'pred']]
    pred_df['truth'] = exact_truth
    pred_df['truth_strain'] = strain_truth

    # calculate for hybrid exact/strain-level matches
    TP = calc_tp(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    FP = calc_fp(pred_df['truth_strain'], pred_df['pred'], pred_df['contig_bp'])
    TN = calc_tn(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    FN = calc_fn(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    str_list = calc_stats(sag_id, 'strain', algo, TP, FP, TN, FN,
                          pred_df['truth_strain'], pred_df['pred']
                          )
    # ALL Recruits
    # calculate for exact-level match
    TP = calc_tp(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    FP = calc_fp(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    TN = calc_tn(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    FN = calc_fn(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    x_list = calc_stats(sag_id, 'exact', algo, TP, FP, TN, FN,
                        pred_df['truth'], pred_df['pred']
                        )
    cat_list = [str_list, x_list]

    return cat_list


def calc_tp(y_truth, y_pred, bp_cnt):
    tp_list = pd.Series([1 if ((x[0] == 1) & (x[1] == 1)) else 0 for x in zip(y_truth, y_pred)])
    tp_bp_list = pd.Series([x[0] * x[1] for x in zip(tp_list, bp_cnt)])
    TP = tp_bp_list.sum()

    return TP


def calc_fp(y_truth, y_pred, bp_cnt):
    fp_list = pd.Series([1 if ((x[0] == -1) & (x[1] == 1)) else 0 for x in zip(y_truth, y_pred)])
    fp_bp_list = pd.Series([x[0] * x[1] for x in zip(fp_list, bp_cnt)])
    FP = fp_bp_list.sum()

    return FP


def calc_tn(y_truth, y_pred, bp_cnt):
    tn_list = pd.Series([1 if ((x[0] == -1) & (x[1] == -1)) else 0 for x in zip(y_truth, y_pred)])
    tn_bp_list = pd.Series([x[0] * x[1] for x in zip(tn_list, bp_cnt)])
    TN = tn_bp_list.sum()

    return TN


def calc_fn(y_truth, y_pred, bp_cnt):
    fn_list = pd.Series([1 if ((x[0] == 1) & (x[1] == -1)) else 0 for x in zip(y_truth, y_pred)])
    fn_bp_list = pd.Series([x[0] * x[1] for x in zip(fn_list, bp_cnt)])
    FN = fn_bp_list.sum()

    return FN


def calc_stats(sag_id, level, algo, TP, FP, TN, FN, y_truth, y_pred):
    precision = TP / (TP + FP)
    sensitivity = TP / (TP + FN)
    N = TN + TP + FN + FP
    S = (TP + FN) / N
    P = (TP + FP) / N
    D = ((S * P) * (1 - S) * (1 - P)) ** (1 / 2)
    if D == 0:
        D = 1
    MCC = ((TP / N) - S * P) / D
    F1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    oc_precision, oc_recall, _ = precision_recall_curve(y_truth, y_pred)
    AUC = auc(oc_recall, oc_precision)
    stat_list = [sag_id, level, algo, precision, sensitivity, MCC, AUC, F1,
                 N, S, P, TP, FP, TN, FN
                 ]

    return stat_list


# Below is to run cross validation for all features table
#################################################
# Inputs
#################################################
minhash_recruits = sys.argv[1]
nmf_recruits = sys.argv[2]
all_recruits = sys.argv[3]
src2contig_file = sys.argv[4]
sag2cami_file = sys.argv[5]
subcontig_file = sys.argv[6]
output_file = sys.argv[7]
output2_file = sys.argv[8]
nthreads = int(sys.argv[9])
#################################################


# setup mapping to CAMI ref genomes
minhash_df = pd.read_csv(minhash_recruits, sep='\t', header=0)
nmf_tetra_df = pd.read_csv(nmf_recruits, sep='\t', header=0)
all_tetra_df = pd.read_csv(all_recruits, sep='\t', header=0)
src2contig_df = pd.read_csv(src2contig_file, header=0, sep='\t')
src2contig_df = src2contig_df[src2contig_df['CAMI_genomeID'].notna()]
sag2cami_df = pd.read_csv(sag2cami_file, header=0, sep='\t')
subcontig_df = pd.read_csv(subcontig_file, sep='\t', header=0)
contig_df = subcontig_df.drop(['subcontig_id'], axis=1).drop_duplicates()
contig_bp_df = contig_df.merge(src2contig_df[['@@SEQUENCEID', 'bp_cnt']].rename(
    columns={'@@SEQUENCEID': 'contig_id'}), on='contig_id', how='left'
)

# setup multithreading pool
pool = multiprocessing.Pool(processes=nthreads)
arg_list = []
for sag_id in tqdm(sag2cami_df['sag_id'].unique()):
    # subset recruit dataframes
    sag_mh_df = minhash_df.loc[minhash_df['sag_id'] == sag_id]
    if sag_mh_df.shape[0] != 0:
        sag_nmf_df = nmf_tetra_df.loc[nmf_tetra_df['sag_id'] == sag_id]
        sag_all_df = all_tetra_df.loc[all_tetra_df['sag_id'] == sag_id]
        # Map Sources/SAGs to Strain IDs
        src_id = list(sag2cami_df.loc[sag2cami_df['sag_id'] == sag_id]['CAMI_genomeID'])[0]
        strain_id = list(src2contig_df.loc[src2contig_df['CAMI_genomeID'] == src_id
                                           ]['strain'])[0]
        src_sub_df = src2contig_df.loc[src2contig_df['CAMI_genomeID'] == src_id]
        strain_sub_df = src2contig_df.loc[src2contig_df['strain'] == strain_id]
        src2contig_list = list(set(src_sub_df['@@SEQUENCEID'].values))
        src2strain_list = list(set(strain_sub_df['@@SEQUENCEID'].values))
        arg_list.append([sag_id, sag_mh_df[['sag_id', 'contig_id']], sag_nmf_df, sag_all_df, contig_bp_df,
                         src2contig_list, src2strain_list
                         ])
    # else:
    #    print(sag_id, ' has no minhash recruits...')

results = pool.imap_unordered(EArecruit, arg_list)
score_list = []
for i, output in tqdm(enumerate(results, 1)):
    score_list.extend(output)
logging.info('\n')
pool.close()
pool.join()
score_df = pd.DataFrame(score_list, columns=['sag_id', 'level', 'algorithm',
                                             'precision', 'sensitivity', 'MCC', 'AUC', 'F1',
                                             'N', 'S', 'P', 'TP', 'FP', 'TN', 'FN'
                                             ])
sort_score_df = score_df.sort_values(['sag_id', 'level', 'algorithm'], ascending=[False, False, True])
sort_score_df.to_csv(output_file, index=False, sep='\t')
group_score_df = sort_score_df.groupby(['level', 'algorithm'])[
    ['precision', 'sensitivity', 'MCC', 'AUC', 'F1']].mean().reset_index()
group_score_df.to_csv(output2_file, index=False, sep='\t')

# best_MCC = sort_score_df['MCC'].iloc[0]
# best_df = score_df.loc[score_df['MCC'] == best_MCC]
# best_df.to_csv(best_output, index=False, sep='\t')


'''
# builds the sag to cami ID mapping file
mh_list = list(minhash_df['sag_id'].unique())
cami_list = [str(x) for x in src2contig_df['CAMI_genomeID'].unique()]
sag2cami_list = []
print('Mapping Sources to Synthetic SAGs...')
for sag_id in mh_list:
    match = difflib.get_close_matches(str(sag_id), cami_list, n=1, cutoff=0)[0]
    m_len = len(match)
    sub_sag_id = sag_id[:m_len]
    if sub_sag_id != match:
        match = difflib.get_close_matches(str(sub_sag_id), cami_list, n=1, cutoff=0)[0]
        if match == sub_sag_id:
            print("PASSED:", sag_id, sub_sag_id, match)
        else:
            m1_len = len(match)
            sub_sag_id = sag_id[:m_len]
            sub_sub_id = sub_sag_id[:m1_len].split('.')[0]
            match = difflib.get_close_matches(str(sub_sub_id), cami_list, n=1, cutoff=0)[0]
    sag2cami_list.append([sag_id, match])
sag2cami_df = pd.DataFrame(sag2cami_list, columns=['sag_id', 'CAMI_genomeID'])
sag2cami_df.to_csv("~/Desktop/test_NMF/sag2cami_map.tsv", index=False, sep='\t')
'''
