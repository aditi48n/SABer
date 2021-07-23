#!/usr/bin/env python

import logging
import multiprocessing
import sys

import pandas as pd
from sklearn import svm
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

pd.options.mode.chained_assignment = None  # default='warn'


def recruitSubs(p):
    sag_id, mh_sag_df, nmf_table, gamma, nu, src2contig_list, src2strain_list = p

    # load nmf file
    nmf_feat_df = pd.read_csv(nmf_table, sep='\t', header=0, index_col='subcontig_id')

    # start ocsvm cross validation analysis
    sag_nmf_df = nmf_feat_df.loc[nmf_feat_df['contig_id'].isin(sag_mh_df['contig_id'])]
    mg_nmf_df = nmf_feat_df.copy()  # .loc[~nmf_feat_df['contig_id'].isin(sag_mh_df['contig_id'])]
    sag_nmf_df.drop(columns=['contig_id'], inplace=True)
    mg_nmf_df.drop(columns=['contig_id'], inplace=True)

    final_pass_df = runOCSVM(sag_nmf_df, mg_nmf_df, sag_id, gamma, nu)

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
    subcontig_id_list = list(merge_recruits_df['subcontig_id'])
    contig_id_list = list(merge_recruits_df['contig_id'])
    exact_truth = list(merge_recruits_df['exact_truth'])
    strain_truth = list(merge_recruits_df['strain_truth'])
    pred = list(merge_recruits_df['pred'])
    stats_lists = recruit_stats([sag_id, gamma, nu, subcontig_id_list, contig_id_list,
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


def recruit_stats(p):
    sag_id, gam, n, subcontig_id_list, contig_id_list, exact_truth, strain_truth, pred = p
    pred_df = pd.DataFrame(zip(subcontig_id_list, contig_id_list, pred),
                           columns=['subcontig_id', 'contig_id', 'pred']
                           )
    pred_df['sag_id'] = sag_id
    pred_df['gamma'] = gam
    pred_df['nu'] = n

    pred_df = pred_df[['sag_id', 'nu', 'gamma', 'subcontig_id', 'contig_id', 'pred']]

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
    # ALL Recruits
    # calculate for hybrid exact/strain-level matches
    TP = calc_tp(pred_df['truth'], pred_df['all_pred'])
    FP = calc_fp(pred_df['truth_strain'], pred_df['all_pred'])
    TN = calc_tn(pred_df['truth'], pred_df['all_pred'])
    FN = calc_fn(pred_df['truth'], pred_df['all_pred'])
    all_str_list = calc_stats(sag_id, 'strain', 'all', gam, n, TP, FP, TN, FN,
                              pred_df['truth_strain'], pred_df['all_pred']
                              )
    # ALL Recruits
    # calculate for exact-level match
    TP = calc_tp(pred_df['truth'], pred_df['all_pred'])
    FP = calc_fp(pred_df['truth'], pred_df['all_pred'])
    TN = calc_tn(pred_df['truth'], pred_df['all_pred'])
    FN = calc_fn(pred_df['truth'], pred_df['all_pred'])
    all_x_list = calc_stats(sag_id, 'exact', 'all', gam, n, TP, FP, TN, FN,
                            pred_df['truth'], pred_df['all_pred']
                            )

    # Majority-Rule Recruits
    # calculate for hybrid exact/strain-level matches
    TP = calc_tp(pred_df['truth'], pred_df['major_pred'])
    FP = calc_fp(pred_df['truth_strain'], pred_df['major_pred'])
    TN = calc_tn(pred_df['truth'], pred_df['major_pred'])
    FN = calc_fn(pred_df['truth'], pred_df['major_pred'])
    maj_str_list = calc_stats(sag_id, 'strain', 'majority', gam, n, TP, FP, TN, FN,
                              pred_df['truth_strain'], pred_df['major_pred']
                              )
    # Majority-Rule Recruits
    # calculate for exact-level match
    TP = calc_tp(pred_df['truth'], pred_df['major_pred'])
    FP = calc_fp(pred_df['truth'], pred_df['major_pred'])
    TN = calc_tn(pred_df['truth'], pred_df['major_pred'])
    FN = calc_fn(pred_df['truth'], pred_df['major_pred'])
    maj_x_list = calc_stats(sag_id, 'exact', 'majority', gam, n, TP, FP, TN, FN,
                            pred_df['truth'], pred_df['major_pred']
                            )
    filter_pred_df = pred_df.loc[pred_df['major_pred'] == 1]

    return all_str_list, all_x_list, maj_str_list, maj_x_list, filter_pred_df


def calc_tp(y_truth, y_pred):
    tp_list = pd.Series([1 if ((x[0] == 1) & (x[1] == 1)) else 0 for x in zip(y_truth, y_pred)])
    TP = tp_list.sum()

    return TP


def calc_fp(y_truth, y_pred):
    fp_list = pd.Series([1 if ((x[0] == -1) & (x[1] == 1)) else 0 for x in zip(y_truth, y_pred)])
    FP = fp_list.sum()

    return FP


def calc_tn(y_truth, y_pred):
    tn_list = pd.Series([1 if ((x[0] == -1) & (x[1] == -1)) else 0 for x in zip(y_truth, y_pred)])
    TN = tn_list.sum()

    return TN


def calc_fn(y_truth, y_pred):
    fn_list = pd.Series([1 if ((x[0] == 1) & (x[1] == -1)) else 0 for x in zip(y_truth, y_pred)])
    FN = fn_list.sum()

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


'''
# build nmf table from tetra CLR transformed relative abund tetra table
tetra_df = pd.read_csv('~/Desktop/test_NMF/CAMI_high_GoldStandardAssembly.tetras.tsv',
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
nmf_feat_df.to_csv('~/Desktop/test_NMF/CAMI_high_GoldStandardAssembly.nmf_trans_40.tsv',
                   sep='\t', index=False
                   )

sys.exit()
'''

#################################################
# Inputs
#################################################
sag_id = sys.argv[1]
minhash_recruits = sys.argv[2]
nmf_dat = sys.argv[3]
nmf_output = sys.argv[4]
best_output = sys.argv[5]
src2sag_file = sys.argv[6]
nthreads = int(sys.argv[7])

# Example:
# python dev_utils/test_NMF.py \
# 1021_F_run134.final.scaffolds.gt1kb.2806 \
# ~/Desktop/test_NMF/CAMI_high_GoldStandardAssembly.201.mhr_trimmed_recruits.tsv \
# ~/Desktop/test_NMF/CAMI_high_GoldStandardAssembly.nmf_trans.tsv \
# ~/Desktop/test_NMF/nmf_preds/1021_F_run134.final.scaffolds.gt1kb.2806.nmf_recruits.tsv \
# ~/Desktop/test_NMF/nmf_preds/1021_F_run134.final.scaffolds.gt1kb.2806.nmf_best.tsv \
# ~/Desktop/test_NMF/src2sag_map.tsv \
# 6
#################################################

minhash_df = pd.read_csv(minhash_recruits, sep='\t', header=0)
sag_mh_df = minhash_df.loc[minhash_df['sag_id'] == sag_id]
if sag_mh_df.shape[0] != 0:

    # setup mapping to CAMI ref genomes
    src2sag_df = pd.read_csv(src2sag_file, header=0, sep='\t')
    src2sag_df = src2sag_df[src2sag_df['CAMI_genomeID'].notna()]
    sag2src_dict = {}
    sag2strain_dict = {}
    for src_id in set(src2sag_df['CAMI_genomeID']):
        if src_id in sag_id:
            if sag_id in sag2src_dict.keys():
                if len(src_id) > len(sag2src_dict[sag_id]):
                    sag2src_dict[sag_id] = src_id
                    strain_id = list(src2sag_df.loc[src2sag_df['CAMI_genomeID'] == src_id]['strain'])[0]
                    sag2strain_dict[sag_id] = strain_id

            else:
                sag2src_dict[sag_id] = src_id
                strain_id = list(src2sag_df.loc[src2sag_df['CAMI_genomeID'] == src_id]['strain'])[0]
                sag2strain_dict[sag_id] = strain_id
    src2contig_df = src2sag_df.loc[src2sag_df['CAMI_genomeID'] == sag2src_dict[sag_id]]
    src2strain_df = src2sag_df.loc[src2sag_df['strain'] == sag2strain_dict[sag_id]]
    src2contig_list = list(set(src2contig_df['@@SEQUENCEID'].values))
    src2strain_list = list(set(src2strain_df['@@SEQUENCEID'].values))

    gamma_range = [10 ** k for k in range(-6, 6)]
    gamma_range.extend(['scale'])
    nu_range = [k / 10 for k in range(1, 10, 1)]

    pool = multiprocessing.Pool(processes=nthreads)
    arg_list = []
    for gam in gamma_range:
        for n in nu_range:
            arg_list.append([sag_id, sag_mh_df, nmf_dat,
                             gam, n, src2contig_list, src2strain_list
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

# OLD Don't use below #
sys.exit()
nmf_mh_df = nmf_feat_df.merge(minhash_df, on='contig_id')
print(nmf_mh_df.head())
print(nmf_mh_df.shape)
sys.exit()
plt.figure(figsize=(20, 12))
contig_ids = np.array(shifted_df.index)
xs = nmf_features[:, 0]
# Select the 1th feature: ys
ys = nmf_features[:, 1]
zs = nmf_features[:, 2]

# Scatter plot
plt.scatter(xs, ys, alpha=0.5)
# Annotate the points
# for x, y, countries in zip(xs, ys, countries):
#    plt.annotate(contig_ids, (x, y), fontsize=10, alpha=0.5)
plt.show()
plt.clf()

plt.scatter(ys, zs, alpha=0.5)
plt.show()
plt.clf()

plt.scatter(xs, zs, alpha=0.5)
plt.show()
plt.clf()

sys.exit()
eurovision = pd.read_csv("~/Desktop/eurovision-2016.csv")
print(eurovision.head())
televote_Rank = eurovision.pivot(index='From country', columns='To country', values='Televote Rank')
# fill NAs by min per country
televote_Rank.fillna(televote_Rank.min(), inplace=True)
print(televote_Rank.head())
print(televote_Rank.shape)

# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components=2)
# Fit the model to televote_Rank
model.fit(televote_Rank)
# Transform the televote_Rank: nmf_features
nmf_features = model.transform(televote_Rank)
# Print the NMF features
nmf_feat_df = pd.DataFrame(nmf_features)
nmf_comp_df = pd.DataFrame(model.components_)
# print(nmf_feat_df.head())
print(nmf_feat_df.shape)
# print(nmf_comp_df.head())
print(nmf_comp_df.shape)

plt.figure(figsize=(20, 12))
countries = np.array(televote_Rank.index)
xs = nmf_features[:, 0]
# Select the 1th feature: ys
ys = nmf_features[:, 1]

# Scatter plot
plt.scatter(xs, ys, alpha=0.5)
# Annotate the points
# for x, y, countries in zip(xs, ys, countries):
#    plt.annotate(countries, (x, y), fontsize=10, alpha=0.5)
plt.show()
