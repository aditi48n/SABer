import logging
import multiprocessing
from functools import reduce

import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm


def EArecruit(p):  # Error Analysis for all recruits per sag
    col_id, temp_id, temp_clust_df, temp_contig_df, temp_src2contig_list, temp_src2strain_list = p
    temp_clust_df['denovo'] = 1
    temp_contig_df[col_id] = temp_id
    df_list = [temp_contig_df, temp_clust_df]
    merge_recruits_df = reduce(lambda left, right: pd.merge(left, right,
                                                            on=[col_id, 'contig_id'],
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
    algo_list = ['denovo']
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


# setup mapping to CAMI ref genomes
cluster_df = pd.read_csv(
    '/home/ryan/Desktop/test_NMF/minhash_features/'
    'CAMI_high_GoldStandardAssembly.denovo_clusters.tsv',
    sep='\t', header=0
)
cluster_trim_df = cluster_df.query('best_label != -1')
src2contig_df = pd.read_csv('/home/ryan/Desktop/test_NMF/src2contig_map.tsv',
                            header=0, sep='\t'
                            )
src2contig_df = src2contig_df.rename(columns={'@@SEQUENCEID': 'contig_id'})
contig_bp_df = src2contig_df[['contig_id', 'bp_cnt']]
clust2src_df = cluster_trim_df.merge(src2contig_df[['contig_id', 'CAMI_genomeID',
                                                    'strain', 'bp_cnt']],
                                     on='contig_id', how='left'
                                     )
# Add taxonomy to each cluster
clust_tax = []
for clust in clust2src_df['best_label'].unique():
    sub_clust_df = clust2src_df.query('best_label == @clust')
    exact_df = sub_clust_df.groupby(['CAMI_genomeID'])['bp_cnt'].sum().reset_index()
    strain_df = sub_clust_df.groupby(['strain'])['bp_cnt'].sum().reset_index()
    ex_label_df = exact_df[exact_df.bp_cnt == exact_df.bp_cnt.max()]['CAMI_genomeID']
    if not ex_label_df.empty:
        exact_label = exact_df[exact_df.bp_cnt == exact_df.bp_cnt.max()
                               ]['CAMI_genomeID'].values[0]
        strain_label = strain_df[strain_df.bp_cnt == strain_df.bp_cnt.max()
                                 ]['strain'].values[0]
        clust_tax.append([clust, exact_label, strain_label])

clust_tax_df = pd.DataFrame(clust_tax, columns=['best_label', 'exact_label', 'strain_label'])
clust2label_df = clust_tax_df.merge(cluster_trim_df, on='best_label', how='left')
clust2contig_df = clust2label_df[['best_label', 'contig_id', 'exact_label', 'strain_label'
                                  ]].drop_duplicates()
# setup multithreading pool
nthreads = 8
pool = multiprocessing.Pool(processes=nthreads)
arg_list = []
for clust in tqdm(clust2contig_df['best_label'].unique()):
    # subset recruit dataframes
    sub_clust_df = clust2contig_df.query('best_label == @clust')
    dedup_clust_df = sub_clust_df[['best_label', 'contig_id']].drop_duplicates()
    # Map Sources/SAGs to Strain IDs
    src_id = sub_clust_df['exact_label'].values[0]
    strain_id = sub_clust_df['strain_label'].values[0]
    src_sub_df = src2contig_df.query('CAMI_genomeID == @src_id')
    strain_sub_df = src2contig_df.query('strain == @strain_id')
    src2contig_list = list(set(src_sub_df['contig_id'].values))
    src2strain_list = list(set(strain_sub_df['contig_id'].values))
    arg_list.append(['best_label', clust, dedup_clust_df, contig_bp_df, src2contig_list,
                     src2strain_list
                     ])

results = pool.imap_unordered(EArecruit, arg_list)
score_list = []
for i, output in tqdm(enumerate(results, 1)):
    score_list.extend(output)
logging.info('\n')
pool.close()
pool.join()

score_df = pd.DataFrame(score_list, columns=['best_label', 'level', 'algorithm',
                                             'precision', 'sensitivity', 'MCC', 'AUC', 'F1',
                                             'N', 'S', 'P', 'TP', 'FP', 'TN', 'FN'
                                             ])
sort_score_df = score_df.sort_values(['best_label', 'level', 'precision', 'sensitivity'],
                                     ascending=[False, False, True, True]
                                     )
score_tax_df = sort_score_df.merge(clust_tax_df, on='best_label', how='left')
score_tax_df['size_bp'] = score_tax_df['TP'] + score_tax_df['FP']
score_tax_df['>20Kb'] = 'No'
score_tax_df.loc[score_tax_df['size_bp'] >= 20000, '>20Kb'] = 'Yes'
score_tax_df['NC_bins'] = 'No'
score_tax_df.loc[(score_tax_df['precision'] >= 0.95) &
                 (score_tax_df['sensitivity'] >= 0.9), 'NC_bins'] = 'Yes'
score_tax_df['MQ_bins'] = 'No'
score_tax_df.loc[(score_tax_df['precision'] >= 0.9) &
                 (score_tax_df['sensitivity'] >= 0.5), 'MQ_bins'] = 'Yes'

stat_mean_df = score_tax_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                     'MQ_bins'])[['precision', 'sensitivity', 'MCC',
                                                  'AUC', 'F1']].mean().reset_index()
cnt_bins_df = score_tax_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                    'MQ_bins']).size().reset_index()
cnt_bins_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                       'bin_cnt'
                       ]
cnt_exact_df = score_tax_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                     'MQ_bins'])[['exact_label']
].nunique().reset_index()
cnt_exact_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                        'genome_cnt'
                        ]
cnt_genos_df = score_tax_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                     'MQ_bins'])[['strain_label']
].nunique().reset_index()
cnt_genos_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                        'strain_cnt'
                        ]
dfs = [stat_mean_df, cnt_bins_df, cnt_exact_df, cnt_genos_df]
stat_df = reduce(lambda left, right: pd.merge(left, right, on=['level',
                                                               'algorithm', '>20Kb',
                                                               'NC_bins', 'MQ_bins'
                                                               ]), dfs
                 )
score_tax_df.to_csv('/home/ryan/Desktop/test_NMF/minhash_features/'
                    'CH.denovo.errstat.tsv', index=False, sep='\t'
                    )
stat_df.to_csv('/home/ryan/Desktop/test_NMF/minhash_features/'
               'CH.denovo.errstat.mean.tsv', index=False, sep='\t'
               )

##################################################################################################
# Trusted Contigs errstats
# setup mapping to CAMI ref genomes
n_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
score_df_list = []
stats_df_list = []
for n in n_list:
    print(n)
    cluster_df = pd.read_csv(
        '/home/ryan/Desktop/test_NMF/minhash_features/'
        'CH.trusted_clusters.' + n + '.tsv',
        sep='\t', header=0
    )
    src2contig_df = pd.read_csv('/home/ryan/Desktop/test_NMF/src2contig_map.tsv',
                                header=0, sep='\t'
                                )
    src2contig_df = src2contig_df.rename(columns={'@@SEQUENCEID': 'contig_id'})
    sag2cami_df = pd.read_csv('/home/ryan/Desktop/test_NMF/sag2cami_map.tsv', header=0, sep='\t')
    sag2contig_df = sag2cami_df.merge(src2contig_df, on='CAMI_genomeID', how='left')

    # setup multithreading pool
    nthreads = 8
    pool = multiprocessing.Pool(processes=nthreads)
    arg_list = []
    for clust in tqdm(cluster_df['sag_id'].unique()):
        # subset recruit dataframes
        sub_clust_df = cluster_df.query('sag_id == @clust')
        dedup_clust_df = sub_clust_df[['sag_id', 'contig_id']].drop_duplicates()
        # Map Sources/SAGs to Strain IDs
        src_id = sag2contig_df.query('sag_id == @clust')['CAMI_genomeID'].values[0]
        strain_id = sag2contig_df.query('sag_id == @clust')['strain'].values[0]
        src_sub_df = src2contig_df.query('CAMI_genomeID == @src_id')
        strain_sub_df = src2contig_df.query('strain == @strain_id')
        src2contig_list = list(set(src_sub_df['contig_id'].values))
        src2strain_list = list(set(strain_sub_df['contig_id'].values))
        arg_list.append(['sag_id', clust, dedup_clust_df, contig_bp_df, src2contig_list,
                         src2strain_list
                         ])

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
    score_df = score_df.merge(sag2cami_df, left_on='sag_id', right_on='sag_id', how='left')
    score_tax_df = score_df.merge(clust2src_df[['CAMI_genomeID', 'strain']].drop_duplicates(),
                                  on='CAMI_genomeID', how='left'
                                  )
    score_tax_df['dataset'] = n
    score_tax_df['size_bp'] = score_tax_df['TP'] + score_tax_df['FP']
    score_tax_df['>20Kb'] = 'No'
    score_tax_df.loc[score_tax_df['size_bp'] >= 20000, '>20Kb'] = 'Yes'
    score_tax_df['NC_bins'] = 'No'
    score_tax_df.loc[(score_tax_df['precision'] >= 0.95) &
                     (score_tax_df['sensitivity'] >= 0.9), 'NC_bins'] = 'Yes'
    score_tax_df['MQ_bins'] = 'No'
    score_tax_df.loc[(score_tax_df['precision'] >= 0.9) &
                     (score_tax_df['sensitivity'] >= 0.5), 'MQ_bins'] = 'Yes'

    sort_score_df = score_tax_df.sort_values(['dataset', 'sag_id', 'level', 'precision',
                                              'sensitivity'],
                                             ascending=[True, False, False, True, True]
                                             )
    score_df_list.append(sort_score_df)
    stat_mean_df = sort_score_df.groupby(['dataset', 'level', 'algorithm', '>20Kb', 'NC_bins',
                                          'MQ_bins'])[['precision', 'sensitivity', 'MCC',
                                                       'AUC', 'F1']].mean().reset_index()
    cnt_bins_df = sort_score_df.groupby(['dataset', 'level', 'algorithm', '>20Kb', 'NC_bins',
                                         'MQ_bins']).size().reset_index()
    cnt_bins_df.columns = ['dataset', 'level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                           'bin_cnt'
                           ]
    cnt_genos_df = sort_score_df.groupby(['dataset', 'level', 'algorithm', '>20Kb', 'NC_bins',
                                          'MQ_bins'])[['CAMI_genomeID']
    ].nunique().reset_index()
    cnt_genos_df.columns = ['dataset', 'level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                            'genome_cnt'
                            ]
    cnt_strain_df = sort_score_df.groupby(['dataset', 'level', 'algorithm', '>20Kb', 'NC_bins',
                                           'MQ_bins'])[['strain']
    ].nunique().reset_index()
    cnt_strain_df.columns = ['dataset', 'level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                             'strain_cnt'
                             ]
    dfs = [stat_mean_df, cnt_bins_df, cnt_genos_df, cnt_strain_df]
    stat_df = reduce(lambda left, right: pd.merge(left, right, on=['dataset', 'level',
                                                                   'algorithm', '>20Kb',
                                                                   'NC_bins', 'MQ_bins'
                                                                   ]), dfs
                     )
    stats_df_list.append(stat_df)

final_score_df = pd.concat(score_df_list)
final_score_df.to_csv('/home/ryan/Desktop/test_NMF/minhash_features/'
                      'CH.trusted_clusters.errstat.tsv', index=False, sep='\t'
                      )
final_stat_df = pd.concat(stats_df_list)
final_stat_df.to_csv('/home/ryan/Desktop/test_NMF/minhash_features/'
                     'CH.trusted_clusters.errstat.mean.tsv', index=False, sep='\t'
                     )

'''
# split up the SAG MHR by genome into 10 sets
mh_recruits_df = pd.read_csv('/home/ryan/Desktop/test_NMF/minhash_features/'
                          'CAMI_high_GoldStandardAssembly.201.mhr_trimmed_recruits.tsv',
                          header=0, sep='\t')
mh_cami_df = mh_recruits_df.merge(sag2cami_df, on='sag_id', how='left')
print(mh_cami_df.head())
split_dat_dict = {0: [], 1: [], 2: [], 3: [], 4: [],
                  5: [], 6: [], 7: [], 8: [], 9: []
                  }
for cami in mh_cami_df['CAMI_genomeID'].unique():
    sub_cami_df = mh_cami_df.query('CAMI_genomeID == @cami')
    sag_list = set(sub_cami_df['sag_id'].unique())
    for i, sag_id in enumerate(sag_list):
        split_dat_dict[i].append(sag_id)

for k, v in split_dat_dict.items():
    sub_mh_df = mh_recruits_df.query('sag_id in @v')
    sub_mh_df.to_csv('/home/ryan/Desktop/test_NMF/minhash_features/'
                     'CH.201.mhr_recruits.' + str(k) + '.tsv', index=False, sep='\t'
                     )
'''
