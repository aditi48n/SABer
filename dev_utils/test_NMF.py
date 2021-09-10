#!/usr/bin/env python

# from sklearn.decomposition import NMF
import sys

import hdbscan
import pandas as pd
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

pd.options.mode.chained_assignment = None  # default='warn'


def recruitSubs(p):
    sag_id, mh_df, nmf_table, cov_table, gamma, nu, \
    src2contig_list, src2strain_list, contig_bp_df = p

    # load nmf file
    nmf_feat_df = pd.read_csv(nmf_table, sep='\t', header=0, index_col='subcontig_id')
    # load covm file
    cov_df = pd.read_csv(cov_table, sep='\t', header=0)
    cov_df.rename(columns={'contigName': 'subcontig_id'}, inplace=True)
    cov_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cov_df['subcontig_id']]
    cov_df.set_index('subcontig_id', inplace=True)

    # start ocsvm cross validation analysis
    sag_mh_df = mh_df.loc[mh_df['sag_id'] == sag_id]
    mg_mh_df = mh_df.loc[mh_df['sag_id'] != sag_id]
    sag_nmf_df = nmf_feat_df.loc[nmf_feat_df['contig_id'].isin(sag_mh_df['contig_id'])]
    mg_nmf_df = nmf_feat_df.loc[nmf_feat_df['contig_id'].isin(mg_mh_df['contig_id'])]
    sag_nmf_df.drop(columns=['contig_id'], inplace=True)
    mg_nmf_df.drop(columns=['contig_id'], inplace=True)
    sag_cov_df = cov_df.loc[cov_df['contig_id'].isin(sag_mh_df['contig_id'])]
    mg_cov_df = cov_df.loc[cov_df['contig_id'].isin(mg_mh_df['contig_id'])]
    sag_cov_df.drop(columns=['contig_id'], inplace=True)
    mg_cov_df.drop(columns=['contig_id'], inplace=True)

    # merge covM and NMF
    sag_join_df = sag_nmf_df.join(sag_cov_df, lsuffix='_nmf', rsuffix='_covm')
    mg_join_df = mg_nmf_df.join(mg_cov_df, lsuffix='_nmf', rsuffix='_covm')
    sag_join_df['contig_id'] = [x.rsplit('_', 1)[0] for x in sag_join_df.index.values]
    mg_join_df['contig_id'] = [x.rsplit('_', 1)[0] for x in mg_join_df.index.values]
    # add minhash
    sag_merge_df = sag_join_df.merge(sag_mh_df, on='contig_id', how='left')
    mg_merge_df = mg_join_df.merge(mg_mh_df, on='contig_id', how='left')
    sag_merge_df.drop(columns=['contig_id', 'sag_id'], inplace=True)
    mg_merge_df.drop(columns=['contig_id', 'sag_id'], inplace=True)
    sag_merge_df.set_index(sag_join_df.index.values, inplace=True)
    mg_merge_df.set_index(mg_join_df.index.values, inplace=True)

    final_pass_df = runOCSVM(sag_merge_df, mg_merge_df, sag_id, gamma, nu)

    complete_df = pd.DataFrame(mg_nmf_df.index.values, columns=['subcontig_id'])
    complete_df['sag_id'] = sag_id
    complete_df['nu'] = nu
    complete_df['gamma'] = gamma
    complete_df['contig_id'] = [x.rsplit('_', 1)[0] for x in mg_nmf_df.index.values]
    merge_recruits_df = pd.merge(complete_df, final_pass_df,
                                 on=['sag_id', 'nu', 'gamma', 'subcontig_id', 'contig_id'],
                                 how='outer'
                                 )
    merge_recruits_df.fillna(-1, inplace=True)
    merge_recruits_df['exact_truth'] = [1 if x in src2contig_list else -1
                                        for x in merge_recruits_df['contig_id']
                                        ]
    merge_recruits_df['strain_truth'] = [1 if x in src2strain_list else -1
                                         for x in merge_recruits_df['contig_id']
                                         ]
    merge_bp_df = merge_recruits_df.merge(contig_bp_df, on='contig_id', how='left')
    subcontig_id_list = list(merge_bp_df['subcontig_id'])
    contig_id_list = list(merge_bp_df['contig_id'])
    contig_bp_list = list(merge_bp_df['bp_cnt'])
    exact_truth = list(merge_bp_df['exact_truth'])
    strain_truth = list(merge_bp_df['strain_truth'])
    pred = list(merge_bp_df['pred'])

    stats_lists = recruit_stats([sag_id, gamma, nu, subcontig_id_list, contig_id_list, contig_bp_list,
                                 exact_truth, strain_truth, pred
                                 ])
    return stats_lists


def runOCSVM(sag_df, mg_df, sag_id, gamma, nu):
    # fit OCSVM
    clf = svm.OneClassSVM(nu=nu, gamma=gamma)
    clf.fit(sag_df.values)
    mg_pred = clf.predict(mg_df.values)
    contig_id_list = [x.rsplit('_', 1)[0] for x in mg_df.index.values]
    pred_df = pd.DataFrame(zip(mg_df.index.values, contig_id_list, mg_pred),
                           columns=['subcontig_id', 'contig_id', 'pred']
                           )
    pred_df['nu'] = nu
    pred_df['gamma'] = gamma
    pred_df['sag_id'] = sag_id
    pred_df = pred_df[['sag_id', 'nu', 'gamma', 'subcontig_id', 'contig_id', 'pred']]

    return pred_df


def runISOF(sag_df, mg_df, sag_id, contam=0, estim=10, max_samp='auto'):
    # fit IsoForest
    clf = IsolationForest(random_state=42, contamination=contam, n_estimators=estim,
                          max_samples=max_samp
                          )
    clf.fit(sag_df.values)
    sag_pred = clf.predict(sag_df.values)
    sag_score = clf.decision_function(sag_df.values)
    sag_pred_df = pd.DataFrame(data=sag_pred, index=sag_df.index.values,
                               columns=['anomaly'])
    sag_pred_df.loc[sag_pred_df['anomaly'] == 1, 'anomaly'] = 0
    sag_pred_df.loc[sag_pred_df['anomaly'] == -1, 'anomaly'] = 1
    sag_pred_df['scores'] = sag_score
    lower_bound, upper_bound = iqr_bounds(sag_pred_df['scores'], k=0.5)

    mg_pred = clf.predict(mg_df.values)
    mg_score = clf.decision_function(mg_df.values)
    contig_id_list = [x.rsplit('_', 1)[0] for x in mg_df.index.values]
    pred_df = pd.DataFrame(zip(mg_df.index.values, contig_id_list, mg_pred),
                           columns=['subcontig_id', 'contig_id', 'anomaly']
                           )
    pred_df.loc[pred_df['anomaly'] == 1, 'anomaly'] = 0
    pred_df.loc[pred_df['anomaly'] == -1, 'anomaly'] = 1
    pred_df['scores'] = mg_score
    pred_df['iqr_anomaly'] = 0
    pred_df['iqr_anomaly'] = (pred_df['scores'] < lower_bound) | \
                             (pred_df['scores'] > upper_bound)
    pred_df['iqr_anomaly'] = pred_df['iqr_anomaly'].astype(int)
    pred_df['sag_id'] = sag_id
    pred_df['pred'] = pred_df['iqr_anomaly'] == 0
    pred_df['pred'] = pred_df['pred'].astype(int)

    return pred_df


def iqr_bounds(scores, k=1.5):
    q1 = scores.quantile(0.25)
    q3 = scores.quantile(0.75)
    iqr = q3 - q1
    lower_bound = (q1 - k * iqr)
    upper_bound = (q3 + k * iqr)
    return lower_bound, upper_bound


def recruit_stats(p):
    sag_id, gam, n, subcontig_id_list, contig_id_list, contig_bp_list, exact_truth, strain_truth, pred = p
    pred_df = pd.DataFrame(zip(subcontig_id_list, contig_id_list, contig_bp_list, pred),
                           columns=['subcontig_id', 'contig_id', 'contig_bp', 'pred']
                           )
    pred_df['sag_id'] = sag_id
    pred_df['gamma'] = gam
    pred_df['nu'] = n

    pred_df = pred_df[['sag_id', 'nu', 'gamma', 'subcontig_id', 'contig_id', 'contig_bp', 'pred']]

    val_perc = pred_df.groupby('contig_id')['pred'].value_counts(
        normalize=True).reset_index(name='precent')
    pos_perc = val_perc.loc[val_perc['pred'] == 1]
    major_df = pos_perc.loc[pos_perc['precent'] >= 0.51]
    major_pred = [1 if x in list(major_df['contig_id']) else -1
                  for x in pred_df['contig_id']
                  ]
    pos_pred_list = list(set(pred_df.loc[pred_df['pred'] == 1]['contig_id']))
    all_pred = [1 if x in pos_pred_list else -1
                for x in pred_df['contig_id']
                ]
    pred_df['all_pred'] = all_pred
    pred_df['major_pred'] = major_pred
    pred_df['truth'] = exact_truth
    pred_df['truth_strain'] = strain_truth
    dedup_pred_df = pred_df.drop_duplicates(subset=['sag_id', 'nu', 'gamma', 'contig_id', 'contig_bp',
                                                    'pred', 'all_pred', 'major_pred', 'truth',
                                                    'truth_strain'])
    # ALL Recruits
    # calculate for hybrid exact/strain-level matches
    TP = calc_tp(dedup_pred_df['truth'], dedup_pred_df['all_pred'], dedup_pred_df['contig_bp'])
    FP = calc_fp(dedup_pred_df['truth_strain'], dedup_pred_df['all_pred'], dedup_pred_df['contig_bp'])
    TN = calc_tn(dedup_pred_df['truth'], dedup_pred_df['all_pred'], dedup_pred_df['contig_bp'])
    FN = calc_fn(dedup_pred_df['truth'], dedup_pred_df['all_pred'], dedup_pred_df['contig_bp'])
    all_str_list = calc_stats(sag_id, 'strain', 'all', gam, n, TP, FP, TN, FN,
                              dedup_pred_df['truth_strain'], dedup_pred_df['all_pred']
                              )
    # ALL Recruits
    # calculate for exact-level match
    TP = calc_tp(dedup_pred_df['truth'], dedup_pred_df['all_pred'], dedup_pred_df['contig_bp'])
    FP = calc_fp(dedup_pred_df['truth'], dedup_pred_df['all_pred'], dedup_pred_df['contig_bp'])
    TN = calc_tn(dedup_pred_df['truth'], dedup_pred_df['all_pred'], dedup_pred_df['contig_bp'])
    FN = calc_fn(dedup_pred_df['truth'], dedup_pred_df['all_pred'], dedup_pred_df['contig_bp'])
    all_x_list = calc_stats(sag_id, 'exact', 'all', gam, n, TP, FP, TN, FN,
                            dedup_pred_df['truth'], dedup_pred_df['all_pred']
                            )

    # Majority-Rule Recruits
    # calculate for hybrid exact/strain-level matches
    TP = calc_tp(dedup_pred_df['truth'], dedup_pred_df['major_pred'], dedup_pred_df['contig_bp'])
    FP = calc_fp(dedup_pred_df['truth_strain'], dedup_pred_df['major_pred'], dedup_pred_df['contig_bp'])
    TN = calc_tn(dedup_pred_df['truth'], dedup_pred_df['major_pred'], dedup_pred_df['contig_bp'])
    FN = calc_fn(dedup_pred_df['truth'], dedup_pred_df['major_pred'], dedup_pred_df['contig_bp'])
    maj_str_list = calc_stats(sag_id, 'strain', 'majority', gam, n, TP, FP, TN, FN,
                              dedup_pred_df['truth_strain'], dedup_pred_df['major_pred']
                              )
    # Majority-Rule Recruits
    # calculate for exact-level match
    TP = calc_tp(dedup_pred_df['truth'], dedup_pred_df['major_pred'], dedup_pred_df['contig_bp'])
    FP = calc_fp(dedup_pred_df['truth'], dedup_pred_df['major_pred'], dedup_pred_df['contig_bp'])
    TN = calc_tn(dedup_pred_df['truth'], dedup_pred_df['major_pred'], dedup_pred_df['contig_bp'])
    FN = calc_fn(dedup_pred_df['truth'], dedup_pred_df['major_pred'], dedup_pred_df['contig_bp'])
    maj_x_list = calc_stats(sag_id, 'exact', 'majority', gam, n, TP, FP, TN, FN,
                            dedup_pred_df['truth'], dedup_pred_df['major_pred']
                            )
    filter_pred_df = dedup_pred_df.loc[dedup_pred_df['major_pred'] == 1]

    return all_str_list, all_x_list, maj_str_list, maj_x_list, filter_pred_df


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


def calc_stats(sag_id, level, include, gam, n, TP, FP, TN, FN, y_truth, y_pred):
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
    stat_list = [sag_id, level, include, gam, n, precision, sensitivity, MCC, AUC, F1,
                 N, S, P, TP, FP, TN, FN
                 ]

    return stat_list


def filter_pred(p_df, perc):
    val_perc = p_df.groupby('contig_id')['pred'].value_counts(
        normalize=True).reset_index(name='precent')
    pos_perc = val_perc.loc[val_perc['pred'] == 1]
    if perc == 0:
        major_df = pos_perc.loc[pos_perc['precent'] != perc][['contig_id', 'pred']]
    else:
        major_df = pos_perc.loc[pos_perc['precent'] >= perc][['contig_id', 'pred']]
    p_merge_df = p_df.merge(major_df, on='contig_id', how='left', suffixes=('', '_major'))
    p_filter_df = p_merge_df.loc[p_merge_df['pred_major'] == 1]

    return p_filter_df


# Build final table for testing
mh_dat = sys.argv[1]
tetra_dat = sys.argv[2]
cov_dat = sys.argv[3]

# load minhash file
minhash_df = pd.read_csv(mh_dat, sep='\t', header=0)

'''
# Convert CovM to UMAP feature table
cov_df = pd.read_csv('~/Desktop/test_NMF/minhash_features/'
                       'CAMI_high_GoldStandardAssembly.covM.scaled.tsv', header=0, sep='\t',
                       index_col='contigName'
                       )
clusterable_embedding = umap.UMAP(n_neighbors=10, min_dist=0.0, n_components=10,
                                  random_state=42,
                                  ).fit_transform(cov_df)
umap_feat_df = pd.DataFrame(clusterable_embedding, index=cov_df.index.values)
umap_feat_df.reset_index(inplace=True)
umap_feat_df.rename(columns={'index': 'subcontig_id'}, inplace=True)
print(umap_feat_df.head())
umap_feat_df.to_csv('~/Desktop/test_NMF/minhash_features/'
                    'CAMI_high_GoldStandardAssembly.covm_umap10.tsv',
                   sep='\t', index=False
                   )
# Convert Tetra to UMAP feature table
tetra_df = pd.read_csv('~/Desktop/test_NMF/minhash_features/'
                       'CAMI_high_GoldStandardAssembly.tetras.tsv', header=0, sep='\t',
                       index_col='contig_id'
                       )
clusterable_embedding = umap.UMAP(n_neighbors=10, min_dist=0.0, n_components=40,
                                  random_state=42,
                                  ).fit_transform(tetra_df)
umap_feat_df = pd.DataFrame(clusterable_embedding, index=tetra_df.index.values)
umap_feat_df.reset_index(inplace=True)
umap_feat_df.rename(columns={'index': 'subcontig_id'}, inplace=True)
print(umap_feat_df.head())
umap_feat_df.to_csv('~/Desktop/test_NMF/minhash_features/'
                    'CAMI_high_GoldStandardAssembly.tetra_umap40.tsv',
                   sep='\t', index=False
                    )
'''
'''
# Convert MinHash into NMF feature table
#mh_ave_df = minhash_df.pivot(index='sag_id', columns='contig_id', values='jacc_sim_avg')
#mh_ave_df.fillna(0.0, inplace=True)
#print(mh_ave_df.shape)
mh_max_df = minhash_df.pivot(index='sag_id', columns='contig_id', values='jacc_sim_max')
mh_max_df.fillna(0.0, inplace=True)
print(mh_max_df.shape)

#mh_jacc_df = mh_ave_df.join(mh_max_df, lsuffix='_ave', rsuffix='_max')
# Create an NMF instance: model
model = NMF(n_components=200)
#model.fit(mh_ave_df.values)
nmf_features = model.fit_transform(mh_max_df.values)
# Print the NMF features
nmf_comp_df = pd.DataFrame(model.components_)
nmf_feat_df = pd.DataFrame(nmf_features)
nmf_feat_df['sag_id'] = mh_max_df.index.values
nmf_merge_df = nmf_feat_df.merge(denovo_map_df, on='sag_id', how='left')
nmf_merge_df.to_csv('~/Desktop/test_NMF/minhash_features/CAMI_high_GoldStandardAssembly.mhr_nmf_max200.tsv',
                   sep='\t', index=False
                   )
'''

# load nmf file
nmf_feat_df = pd.read_csv(tetra_dat, sep='\t', header=0, index_col='subcontig_id')
nmf_feat_df['contig_id'] = [x.rsplit('_', 1)[0] for x in nmf_feat_df.index.values]
# load covm file
cov_df = pd.read_csv(cov_dat, sep='\t', header=0)
cov_df.rename(columns={'contigName': 'subcontig_id'}, inplace=True)
cov_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cov_df['subcontig_id']]
cov_df.set_index('subcontig_id', inplace=True)
merge_df = nmf_feat_df.join(cov_df, lsuffix='_nmf', rsuffix='_covm')
merge_df.drop(columns=['contig_id_nmf', 'contig_id_covm'], inplace=True)
nmf_feat_df.drop(columns=['contig_id'], inplace=True)
cov_df.drop(columns=['contig_id'], inplace=True)

'''
shifted_df = cov_df + -cov_df.min().min()
# Create an NMF instance: model
model = NMF(n_components=3)
model.fit(shifted_df)
cov_features = model.transform(shifted_df)
# Print the NMF features
cov_comp_df = pd.DataFrame(model.components_)
cov_feat_df = pd.DataFrame(cov_features)
cov_feat_df.set_index(shifted_df.index.values, inplace=True)
'''

clusterer = hdbscan.HDBSCAN(min_cluster_size=200, cluster_selection_method='eom',
                            prediction_data=True, cluster_selection_epsilon=0,
                            min_samples=200
                            # ,allow_single_cluster=True
                            ).fit(merge_df.values)
cluster_labels = clusterer.labels_
cluster_probs = clusterer.probabilities_
cluster_outlier = clusterer.outlier_scores_

cluster_df = pd.DataFrame(zip(merge_df.index.values, cluster_labels, cluster_probs,
                              cluster_outlier),
                          columns=['subcontig_id', 'label', 'probabilities',
                                   'outlier_score']
                          )
cluster_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cluster_df['subcontig_id']]
print(cluster_df.head())
cluster_df.to_csv('/home/ryan/Desktop/test_NMF/minhash_features/'
                  'CAMI_high_GoldStandardAssembly.hdbscan.tsv',
                  sep='\t', index=False
                  )
'''
cluster_df = pd.read_csv('/home/ryan/Desktop/test_NMF/minhash_features/'
                         'CAMI_high_GoldStandardAssembly.hdbscan.tsv',
                         header=0, sep='\t'
                         )
'''
ns_ratio_list = []
for contig in cluster_df['contig_id'].unique():
    sub_df = cluster_df.query('contig_id == @contig')
    noise_cnt = sub_df.query('label == -1').shape[0]
    signal_cnt = sub_df.query('label != -1').shape[0]
    ns_ratio = (noise_cnt / (noise_cnt + signal_cnt)) * 100
    prob_df = sub_df.groupby(['label'])['probabilities'].mean().reset_index().set_index('label')
    best_label = prob_df.idxmax().values[0]
    print(contig, ns_ratio, best_label)
    ns_ratio_list.append([contig, noise_cnt, signal_cnt, ns_ratio, best_label])
ns_ratio_df = pd.DataFrame(ns_ratio_list, columns=['contig_id', 'noise', 'signal', 'ns_ratio', 'best_label'])
cluster_ns_df = cluster_df.merge(ns_ratio_df, on='contig_id', how='left')
no_noise_df = cluster_ns_df.query('ns_ratio < 51')
noise_df = cluster_ns_df.query('ns_ratio >= 51')

print('noise subcontigs:', noise_df.shape[0])
print('cluster subcontigs:', no_noise_df.shape[0])
print('noise contigs:', len(noise_df['contig_id'].unique()))
print('cluster contigs:', len(no_noise_df['contig_id'].unique()))
print('noise clusters:', len(noise_df['label'].unique()))
print('cluster clusters:', len(no_noise_df['label'].unique()))
print('noise best clusters:', len(noise_df['label'].unique()))
print('cluster best clusters:', len(no_noise_df['label'].unique()))

no_noise_df.to_csv('/home/ryan/Desktop/test_NMF/minhash_features/'
                   'CAMI_high_GoldStandardAssembly.no_noise.tsv',
                   sep='\t', index=False
                   )
sys.exit()
denovo_map_df = pd.read_csv('~/Desktop/test_NMF/minhash_features/denovo_id_map.tsv',
                            header=0, sep='\t'
                            )
mh_map_df = minhash_df.merge(denovo_map_df, on='sag_id', how='left')
mh_map_df.columns = ['sag_id', 'contig_id_query', 'subcontig_recruits', 'jacc_sim_avg',
                     'jacc_sim_max', 'contig_id'
                     ]
covm_tetra_feat_df = merge_df.copy()
covm_tetra_feat_df['contig_id'] = [x.rsplit('_', 1)[0] for x in covm_tetra_feat_df.index.values]
subclust_list = []
for cluster in no_noise_df['label'].unique():
    if cluster != -1:
        sub_clust_df = no_noise_df.query('label == @cluster')
        clust_contigs = list(sub_clust_df['contig_id'].unique())
        sub_mh_df = mh_map_df.query(
            'contig_id in @clust_contigs and contig_id_query in @clust_contigs'
        )
        sub_covm_tetra_df = covm_tetra_feat_df.query('contig_id in @clust_contigs')
        # Convert MinHash to UMAP feature table
        mh_ave_df = sub_mh_df.pivot(index='contig_id', columns='contig_id_query',
                                    values='jacc_sim_avg'
                                    )
        mh_ave_df.fillna(0.0, inplace=True)
        mh_max_df = sub_mh_df.pivot(index='contig_id', columns='contig_id_query',
                                    values='jacc_sim_max'
                                    )
        mh_max_df.fillna(0.0, inplace=True)
        mh_jacc_df = mh_ave_df.join(mh_max_df, lsuffix='_ave', rsuffix='_max')
        n_comp = 25
        n_neigh = 25
        p_init = 'spectral'
        if n_comp >= mh_jacc_df.shape[0] - 1:
            p_init = 'random'
            n_neigh = mh_jacc_df.shape[0] - 1
            if n_neigh <= 1:
                n_neigh = 2
        try:
            clusterable_embedding = umap.UMAP(n_neighbors=n_neigh, min_dist=0.0,
                                              n_components=n_comp,
                                              random_state=42, init=p_init, densmap=True
                                              ).fit_transform(mh_jacc_df)
            umap_feat_df = pd.DataFrame(clusterable_embedding, index=mh_jacc_df.index.values)
            umap_feat_df['contig_id'] = [x.rsplit('_', 1)[0] for x in umap_feat_df.index.values]
            mh_covm_tetra_df = sub_covm_tetra_df.merge(umap_feat_df, on='contig_id', how='left'
                                                       ).set_index(sub_covm_tetra_df.index.values)
            mh_covm_tetra_df.drop(columns=['contig_id'], inplace=True)
            mh_covm_tetra_df.fillna(0.0, inplace=True)

            c_size = 100
            if c_size >= mh_covm_tetra_df.shape[0]:
                c_size = mh_covm_tetra_df.shape[0]
            clusterer = hdbscan.HDBSCAN(min_cluster_size=c_size, cluster_selection_method='eom',
                                        prediction_data=True, cluster_selection_epsilon=0,
                                        min_samples=c_size
                                        # ,allow_single_cluster=True
                                        ).fit(mh_covm_tetra_df.values)
            cluster_labels = clusterer.labels_
            cluster_probs = clusterer.probabilities_
            cluster_outlier = clusterer.outlier_scores_
            subclust_df = pd.DataFrame(zip(mh_covm_tetra_df.index.values, cluster_labels,
                                           cluster_probs, cluster_outlier),
                                       columns=['subcontig_id', 'clean_label', 'probabilities',
                                                'outlier_score']
                                       )
            subclust_df['label'] = cluster
            subclust_df['contig_id'] = [x.rsplit('_', 1)[0] for x in subclust_df['subcontig_id']]
        except:
            subclust_df = sub_clust_df
            subclust_df['clean_label'] = cluster
        subclust_list.append(subclust_df)
        print(cluster, len(subclust_df['clean_label'].unique()),
              subclust_df['clean_label'].unique()
              )

final_subclust_df = pd.concat(subclust_list)
print(final_subclust_df.head())
final_subclust_df.to_csv('/home/ryan/Desktop/test_NMF/minhash_features/'
                         'CAMI_high_GoldStandardAssembly.cleaned_clust.tsv',
                         sep='\t', index=False
                         )

'''
# Build superclusters
merge_clust_list = []
tmp_cluster_df = cluster_df.query('label != -1').copy()
for clust in cluster_df['label'].unique():
    clust_list = [clust]
    if clust in tmp_cluster_df['label'].unique():
        for clust2 in cluster_df['label'].unique():
            if ((clust in tmp_cluster_df['label'].unique()) & (clust != clust2)):
                sub_clust_df = tmp_cluster_df.query('label == @clust')
                sub_clust2_df = tmp_cluster_df.query('label == @clust2')
                contig_set = set(sub_clust_df['contig_id'].unique())
                contig2_set = set(sub_clust2_df['contig_id'].unique())
                intersect = contig_set.intersection(contig2_set)
                union = contig_set.union(contig2_set)
                jaccard = len(intersect)/len(union)
                if jaccard >= 0.1:
                    print(clust, clust2, len(intersect), jaccard)
                    clust_list.append(clust2)
        merge_clust_list.append(clust_list)
    tmp_cluster_df = tmp_cluster_df.query('label not in @clust_list')

df_list = []
for i, l in enumerate(merge_clust_list):
    sub_l_df = cluster_df.query('label in @l')
    sub_l_df['supercluster'] = i
    df_list.append(sub_l_df)

superclust_df = pd.concat(df_list)
print(superclust_df.head())
superclust_df.to_csv('/home/ryan/Desktop/test_NMF/minhash_features/'
                     'CAMI_high_GoldStandardAssembly.supercluster.tsv', sep='\t', index=False
                     )
'''

sys.exit()
pred_df_dict = {'ocsvm': [], 'isof': [], 'inter': []}
for sag_id in minhash_df['sag_id'].unique():
    print(sag_id)
    # subset all the DFs
    sag_mh_df = minhash_df.loc[minhash_df['sag_id'] == sag_id]
    mg_mh_df = minhash_df.loc[minhash_df['sag_id'] != sag_id]
    sag_nmf_df = nmf_feat_df.loc[nmf_feat_df['contig_id'].isin(sag_mh_df['contig_id'])]
    mg_nmf_df = nmf_feat_df.loc[nmf_feat_df['contig_id'].isin(mg_mh_df['contig_id'])]
    sag_nmf_df.drop(columns=['contig_id'], inplace=True)
    mg_nmf_df.drop(columns=['contig_id'], inplace=True)
    sag_cov_df = cov_df.loc[cov_df['contig_id'].isin(sag_mh_df['contig_id'])]
    mg_cov_df = cov_df.loc[cov_df['contig_id'].isin(mg_mh_df['contig_id'])]
    sag_cov_df.drop(columns=['contig_id'], inplace=True)
    mg_cov_df.drop(columns=['contig_id'], inplace=True)
    # merge covM and NMF
    sag_join_df = sag_nmf_df.join(sag_cov_df, lsuffix='_nmf', rsuffix='_covm')
    mg_join_df = mg_nmf_df.join(mg_cov_df, lsuffix='_nmf', rsuffix='_covm')
    sag_join_df['contig_id'] = [x.rsplit('_', 1)[0] for x in sag_join_df.index.values]
    mg_join_df['contig_id'] = [x.rsplit('_', 1)[0] for x in mg_join_df.index.values]
    # add minhash
    sag_merge_df = sag_join_df.merge(sag_mh_df, on='contig_id', how='left')
    mg_merge_df = mg_join_df.merge(mg_mh_df, on='contig_id', how='left')
    sag_merge_df.drop(columns=['contig_id', 'sag_id'], inplace=True)
    mg_merge_df.drop(columns=['contig_id', 'sag_id'], inplace=True)
    sag_merge_df.set_index(sag_join_df.index.values, inplace=True)
    mg_merge_df.set_index(mg_join_df.index.values, inplace=True)
    # only run non-singletons
    if sag_merge_df.shape[0] > 1:
        # start ocsvm cross validation analysis
        ocsvm_pred_df = runOCSVM(sag_merge_df, mg_merge_df, sag_id, 1e-06, 0.6)
        ocsvm_filter_df = filter_pred(ocsvm_pred_df, 0.51)
        # start isof analysis
        isof_pred_df = runISOF(sag_merge_df, mg_merge_df, sag_id)
        isof_filter_df = filter_pred(isof_pred_df, 0.99)
        ocsvm_merge_df = pd.concat([sag_mh_df[['sag_id', 'contig_id']],
                                    ocsvm_filter_df[['sag_id', 'contig_id']]]
                                   ).drop_duplicates()
        isof_merge_df = pd.concat([sag_mh_df[['sag_id', 'contig_id']],
                                   isof_filter_df[['sag_id', 'contig_id']]]
                                  ).drop_duplicates()
        diff = set(ocsvm_merge_df['contig_id']).symmetric_difference(set(isof_merge_df['contig_id']))
        inter = set(ocsvm_merge_df['contig_id']).intersection(set(isof_merge_df['contig_id']))
        print('OC-SVM: Recruited', ocsvm_filter_df.shape[0], 'subcontigs...')
        print('OC-SVM: Total of', ocsvm_filter_df[['sag_id', 'contig_id']].drop_duplicates().shape[0],
              'contigs...')
        print('OC-SVM: Total of', ocsvm_merge_df.shape[0], 'contigs with minhash...')
        print('IsoF: Recruited', isof_filter_df.shape[0], 'subcontigs...')
        print('IsoF: Total of', isof_filter_df[['sag_id', 'contig_id']].drop_duplicates().shape[0],
              'contigs...')
        print('IsoF: Total of', isof_merge_df.shape[0], 'contigs with minhash...')
        print('Difference:', len(diff), 'contigs...')
        print('Intersection:', len(inter), 'contigs...')
        print('MinHash:', sag_mh_df.shape[0], 'contigs...')
        pred_df_dict['ocsvm'].append(ocsvm_merge_df)
        pred_df_dict['isof'].append(isof_merge_df)
        # Intersection of the two methods
        inter_merge_df = ocsvm_merge_df.loc[ocsvm_merge_df['contig_id'].isin(inter)]
        pred_df_dict['inter'].append(inter_merge_df)

ocsvm_final_df = pd.concat(pred_df_dict['ocsvm'])
ocsvm_final_df.to_csv('~/Desktop/test_NMF/minhash_features/CAMI_high_GoldStandardAssembly.nmf_ocsvm_recruits.tsv',
                      sep='\t', index=False
                      )
isof_final_df = pd.concat(pred_df_dict['isof'])
isof_final_df.to_csv('~/Desktop/test_NMF/minhash_features/CAMI_high_GoldStandardAssembly.nmf_isof_recruits.tsv',
                     sep='\t', index=False
                     )
inter_final_df = pd.concat(pred_df_dict['inter'])
inter_final_df.to_csv('~/Desktop/test_NMF/minhash_features/CAMI_high_GoldStandardAssembly.nmf_inter_recruits.tsv',
                      sep='\t', index=False
                      )

sys.exit()

# Below is to run cross validation for  covM abundance and nmf tetra
#################################################
# Inputs
#################################################
sag_id = sys.argv[1]
mh_dat = sys.argv[2]
tetra_dat = sys.argv[3]
cov_dat = sys.argv[4]
nmf_output = sys.argv[5]
best_output = sys.argv[6]
src2contig_file = sys.argv[7]
sag2cami_file = sys.argv[8]
subcontig_file = sys.argv[9]
denov_map = sys.argv[10]
nthreads = int(sys.argv[11])

# Example:
# python
# dev_utils/test_NMF.py
# CH_30130
# ~/Desktop/test_NMF/minhash_features/CAMI_high_GoldStandardAssembly.mhr_nmf_20.tsv
# ~/Desktop/test_NMF/minhash_features/CAMI_high_GoldStandardAssembly.tetra_nmf_20.tsv
# ~/Desktop/test_NMF/minhash_features/CAMI_high_GoldStandardAssembly.covM.scaled.tsv
# ~/Desktop/test_NMF/minhash_features/nmf_preds/CH_30130.nmf_scores.tsv
# ~/Desktop/test_NMF/minhash_features/nmf_preds/CH_30130.nmf_best.tsv
# ~/Desktop/test_NMF/src2contig_map.tsv ~/Desktop/test_NMF/sag2cami_map.tsv
# ~/Desktop/test_NMF/subcontig_list.tsv ~/Desktop/test_NMF/minhash_features/denovo_id_map.tsv
# 6

#################################################

# setup mapping to CAMI ref genomes
minhash_df = pd.read_csv(mh_dat, sep='\t', header=0)
src2contig_df = pd.read_csv(src2contig_file, header=0, sep='\t')
src2contig_df = src2contig_df[src2contig_df['CAMI_genomeID'].notna()]
sag2cami_df = pd.read_csv(sag2cami_file, header=0, sep='\t')
subcontig_df = pd.read_csv(subcontig_file, sep='\t', header=0)
contig_df = subcontig_df.drop(['subcontig_id'], axis=1).drop_duplicates()
contig_bp_df = contig_df.merge(src2contig_df[['@@SEQUENCEID', 'bp_cnt']].rename(
    columns={'@@SEQUENCEID': 'contig_id'}), on='contig_id', how='left'
)
denovo_map_df = pd.read_csv(denov_map, header=0, sep='\t')

sag_mh_df = minhash_df.loc[minhash_df['sag_id'] == sag_id]
if sag_mh_df.shape[0] != 0:
    # Map Sources/SAGs to Strain IDs
    dev_id = list(denovo_map_df.loc[denovo_map_df['sag_id'] == sag_id
                                    ]['contig_id'])[0]
    src_id = list(src2contig_df.loc[src2contig_df['@@SEQUENCEID'] == dev_id
                                    ]['CAMI_genomeID'])[0]
    strain_id = list(src2contig_df.loc[src2contig_df['CAMI_genomeID'] == src_id
                                       ]['strain'])[0]
    src_sub_df = src2contig_df.loc[src2contig_df['CAMI_genomeID'] == src_id]
    strain_sub_df = src2contig_df.loc[src2contig_df['strain'] == strain_id]
    src2contig_list = list(set(src_sub_df['@@SEQUENCEID'].values))
    src2strain_list = list(set(strain_sub_df['@@SEQUENCEID'].values))
    print(sag_id, src_id, strain_id)
    gamma_range = [10 ** k for k in range(-6, 6)]
    gamma_range.extend(['scale'])
    nu_range = [k / 10 for k in range(1, 10, 1)]

    pool = multiprocessing.Pool(processes=nthreads)
    arg_list = []
    for gam in gamma_range:
        for n in nu_range:
            arg_list.append([sag_id, minhash_df, tetra_dat, cov_dat,
                             gam, n, src2contig_list, src2strain_list, contig_bp_df
                             ])
    results = pool.imap_unordered(recruitSubs, arg_list)
    score_list = []
    for i, output in enumerate(results, 1):
        print('\rRecruiting with NMF Tetra Model: {}/{}'.format(i, len(arg_list)))
        score_list.append(output[0])
        score_list.append(output[1])
        score_list.append(output[2])
        score_list.append(output[3])
    logging.info('\n')
    pool.close()
    pool.join()
    score_df = pd.DataFrame(score_list, columns=['sag_id', 'level', 'inclusion', 'gamma', 'nu',
                                                 'precision', 'sensitivity', 'MCC', 'AUC', 'F1',
                                                 'N', 'S', 'P', 'TP', 'FP', 'TN', 'FN'
                                                 ])
    score_df.to_csv(nmf_output, index=False, sep='\t')
    sort_score_df = score_df.sort_values(['MCC'], ascending=[False])
    best_MCC = sort_score_df['MCC'].iloc[0]
    best_df = score_df.loc[score_df['MCC'] == best_MCC]
    best_df.to_csv(best_output, index=False, sep='\t')
else:
    print(sag_id, ' has no minhash recruits...')

'''
# Build final table for testing
minhash_recruits = sys.argv[1]
nmf_dat = sys.argv[2]
cov_dat = sys.argv[3]

# load minhash file
minhash_df = pd.read_csv(minhash_recruits, sep='\t', header=0)
# load nmf file
nmf_feat_df = pd.read_csv(nmf_dat, sep='\t', header=0, index_col='subcontig_id')
# load covm file
cov_df = pd.read_csv(cov_dat, sep='\t', header=0)
cov_df.rename(columns={'contigName': 'subcontig_id'}, inplace=True)
cov_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cov_df['subcontig_id']]
cov_df.set_index('subcontig_id', inplace=True)

pred_df_dict = {'ocsvm': [], 'isof': [], 'inter': []}
for sag_id in minhash_df['sag_id'].unique():
    print(sag_id)
    mh_sag_df = minhash_df.loc[minhash_df['sag_id'] == sag_id]
    sag_nmf_df = nmf_feat_df.loc[nmf_feat_df['contig_id'].isin(mh_sag_df['contig_id'])]
    mg_nmf_df = nmf_feat_df.loc[~nmf_feat_df['contig_id'].isin(mh_sag_df['contig_id'])]
    sag_nmf_df.drop(columns=['contig_id'], inplace=True)
    mg_nmf_df.drop(columns=['contig_id'], inplace=True)
    sag_cov_df = cov_df.loc[cov_df['contig_id'].isin(mh_sag_df['contig_id'])]
    mg_cov_df = cov_df.loc[~cov_df['contig_id'].isin(mh_sag_df['contig_id'])]
    sag_cov_df.drop(columns=['contig_id'], inplace=True)
    mg_cov_df.drop(columns=['contig_id'], inplace=True)
    # merge covM and NMF
    sag_join_df = sag_nmf_df.join(sag_cov_df, lsuffix='_nmf', rsuffix='_covm')
    mg_join_df = mg_nmf_df.join(mg_cov_df, lsuffix='_nmf', rsuffix='_covm')
    # start ocsvm analysis
    ocsvm_pred_df = runOCSVM(sag_join_df, mg_join_df, sag_id, 10, 0.1)
    # start isof analysis
    isof_pred_df = runISOF(sag_join_df, mg_join_df, sag_id)
    ocsvm_filter_df = filter_pred(ocsvm_pred_df, 0.51)
    isof_filter_df = filter_pred(isof_pred_df, 0.51)
    ocsvm_merge_df = pd.concat([mh_sag_df[['sag_id', 'contig_id']],
                                ocsvm_filter_df[['sag_id', 'contig_id']]]
                               ).drop_duplicates()
    isof_merge_df = pd.concat([mh_sag_df[['sag_id', 'contig_id']],
                               isof_filter_df[['sag_id', 'contig_id']]]
                              ).drop_duplicates()
    diff = set(ocsvm_merge_df['contig_id']).symmetric_difference(set(isof_merge_df['contig_id']))
    inter = set(ocsvm_merge_df['contig_id']).intersection(set(isof_merge_df['contig_id']))
    print('OC-SVM: Recruited', ocsvm_filter_df.shape[0], 'subcontigs...')
    print('OC-SVM: Total of', ocsvm_filter_df[['sag_id', 'contig_id']].drop_duplicates().shape[0],
          'contigs...')
    print('OC-SVM: Total of', ocsvm_merge_df.shape[0], 'contigs with minhash...')
    print('IsoF: Recruited', isof_filter_df.shape[0], 'subcontigs...')
    print('IsoF: Total of', isof_filter_df[['sag_id', 'contig_id']].drop_duplicates().shape[0],
          'contigs...')
    print('IsoF: Total of', isof_merge_df.shape[0], 'contigs with minhash...')
    print('Difference:', len(diff), 'contigs...')
    print('Intersection:', len(inter), 'contigs...')
    print('MinHash:', mh_sag_df.shape[0], 'contigs...')
    pred_df_dict['ocsvm'].append(ocsvm_merge_df)
    pred_df_dict['isof'].append(isof_merge_df)
    # Intersection of the two methods
    inter_merge_df = ocsvm_merge_df.loc[ocsvm_merge_df['contig_id'].isin(inter)]
    pred_df_dict['inter'].append(inter_merge_df)

ocsvm_final_df = pd.concat(pred_df_dict['ocsvm'])
ocsvm_final_df.to_csv('~/Desktop/test_NMF/CAMI_high_GoldStandardAssembly.nmf_ocsvm_recruits.tsv',
                      sep='\t', index=False
                      )
isof_final_df = pd.concat(pred_df_dict['isof'])
isof_final_df.to_csv('~/Desktop/test_NMF/CAMI_high_GoldStandardAssembly.nmf_isof_recruits.tsv',
                     sep='\t', index=False
                     )
inter_final_df = pd.concat(pred_df_dict['inter'])
inter_final_df.to_csv('~/Desktop/test_NMF/CAMI_high_GoldStandardAssembly.nmf_inter_recruits.tsv',
                      sep='\t', index=False
                      )
sys.exit()
'''
'''
# Below is to run cross validation for covM abundance and nmf tetra
#################################################
# Inputs
#################################################
sag_id = sys.argv[1]
minhash_recruits = sys.argv[2]
nmf_dat = sys.argv[3]
cov_dat = sys.argv[4]
nmf_output = sys.argv[5]
best_output = sys.argv[6]
src2contig_file = sys.argv[7]
sag2cami_file = sys.argv[8]
subcontig_file = sys.argv[9]
nthreads = int(sys.argv[10])

# Example:
# python
# dev_utils/test_NMF.py
# 1021_F_run134.final.scaffolds.gt1kb.2806
# ~/Desktop/test_NMF/CAMI_high_GoldStandardAssembly.201.mhr_trimmed_recruits.tsv
# ~/Desktop/test_NMF/CAMI_high_GoldStandardAssembly.nmf_trans_20.tsv
# ~/Desktop/test_NMF/CAMI_high_GoldStandardAssembly.covM.scaled.tsv
# ~/Desktop/test_NMF/nmf_preds/1021_F_run134.final.scaffolds.gt1kb.2806.nmf_scores.tsv
# ~/Desktop/test_NMF/nmf_preds/1021_F_run134.final.scaffolds.gt1kb.2806.nmf_best.tsv
# ~/Desktop/test_NMF/src2sag_map.tsv
# 6

#################################################

# setup mapping to CAMI ref genomes
minhash_df = pd.read_csv(minhash_recruits, sep='\t', header=0)
src2contig_df = pd.read_csv(src2contig_file, header=0, sep='\t')
src2contig_df = src2contig_df[src2contig_df['CAMI_genomeID'].notna()]
sag2cami_df = pd.read_csv(sag2cami_file, header=0, sep='\t')
subcontig_df = pd.read_csv(subcontig_file, sep='\t', header=0)
contig_df = subcontig_df.drop(['subcontig_id'], axis=1).drop_duplicates()
contig_bp_df = contig_df.merge(src2contig_df[['@@SEQUENCEID', 'bp_cnt']].rename(
    columns={'@@SEQUENCEID': 'contig_id'}), on='contig_id', how='left'
)

sag_mh_df = minhash_df.loc[minhash_df['sag_id'] == sag_id]
if sag_mh_df.shape[0] != 0:
    # Map Sources/SAGs to Strain IDs
    src_id = list(sag2cami_df.loc[sag2cami_df['sag_id'] == sag_id]['CAMI_genomeID'])[0]
    strain_id = list(src2contig_df.loc[src2contig_df['CAMI_genomeID'] == src_id
                                       ]['strain'])[0]
    src_sub_df = src2contig_df.loc[src2contig_df['CAMI_genomeID'] == src_id]
    strain_sub_df = src2contig_df.loc[src2contig_df['strain'] == strain_id]
    src2contig_list = list(set(src_sub_df['@@SEQUENCEID'].values))
    src2strain_list = list(set(strain_sub_df['@@SEQUENCEID'].values))
    print(sag_id, src_id, strain_id)

    gamma_range = [10 ** k for k in range(-6, 6)]
    gamma_range.extend(['scale'])
    nu_range = [k / 10 for k in range(1, 10, 1)]

    pool = multiprocessing.Pool(processes=nthreads)
    arg_list = []
    for gam in gamma_range:
        for n in nu_range:
            arg_list.append([sag_id, sag_mh_df, nmf_dat, cov_dat,
                             gam, n, src2contig_list, src2strain_list, contig_bp_df
                             ])
    results = pool.imap_unordered(recruitSubs, arg_list)
    score_list = []
    for i, output in enumerate(results, 1):
        print('\rRecruiting with NMF Tetra Model: {}/{}'.format(i, len(arg_list)))
        score_list.append(output[0])
        score_list.append(output[1])
        score_list.append(output[2])
        score_list.append(output[3])
    logging.info('\n')
    pool.close()
    pool.join()
    score_df = pd.DataFrame(score_list, columns=['sag_id', 'level', 'inclusion', 'gamma', 'nu',
                                                 'precision', 'sensitivity', 'MCC', 'AUC', 'F1',
                                                 'N', 'S', 'P', 'TP', 'FP', 'TN', 'FN'
                                                 ])
    score_df.to_csv(nmf_output, index=False, sep='\t')
    sort_score_df = score_df.sort_values(['MCC'], ascending=[False])
    best_MCC = sort_score_df['MCC'].iloc[0]
    best_df = score_df.loc[score_df['MCC'] == best_MCC]
    best_df.to_csv(best_output, index=False, sep='\t')
else:
    print(sag_id, ' has no minhash recruits...')
'''
'''
# Build NMF tables
denovo_map_df = pd.read_csv('~/Desktop/test_NMF/minhash_features/denovo_id_map.tsv', header=0,
                            sep='\t'
                            )
# Convert MinHash into NMF feature table
mh_piv_df = minhash_df.pivot(index='sag_id', columns='contig_id', values='jacc_sim_avg')
mh_piv_df.fillna(0.0, inplace=True)
# Create an NMF instance: model
model = NMF(n_components=20)
model.fit(mh_piv_df)
nmf_features = model.transform(mh_piv_df)
# Print the NMF features
nmf_comp_df = pd.DataFrame(model.components_)
nmf_feat_df = pd.DataFrame(nmf_features)
nmf_feat_df['sag_id'] = mh_piv_df.index.values
nmf_merge_df = nmf_feat_df.merge(denovo_map_df, on='sag_id', how='left')
nmf_merge_df.to_csv('~/Desktop/test_NMF/minhash_features/CAMI_high_GoldStandardAssembly.mhr_nmf_20.tsv',
                   sep='\t', index=False
                   )

# build nmf table from tetra CLR transformed relative abund tetra table
tetra_df = pd.read_csv('~/Desktop/test_NMF/minhash_features/CAMI_high_GoldStandardAssembly.tetras.tsv',
                       sep='\t', header=0, index_col='contig_id'
                       )
shifted_df = tetra_df + -tetra_df.min().min()
# Create an NMF instance: model
model = NMF(n_components=20)
model.fit(shifted_df)
nmf_features = model.transform(shifted_df)
# Print the NMF features
nmf_comp_df = pd.DataFrame(model.components_)
nmf_feat_df = pd.DataFrame(nmf_features)
nmf_feat_df['subcontig_id'] = index = shifted_df.index.values
nmf_feat_df['contig_id'] = [x.rsplit('_', 1)[0] for x in nmf_feat_df['subcontig_id']]
nmf_feat_df.to_csv('~/Desktop/test_NMF/minhash_features/CAMI_high_GoldStandardAssembly.tetra_nmf_20.tsv',
                   sep='\t', index=False
                   )

sys.exit()
'''
