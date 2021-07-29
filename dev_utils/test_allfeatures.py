#!/usr/bin/env python

import logging
import multiprocessing
import sys

import pandas as pd

# Below is to run cross validation for all features table
#################################################
# Inputs
#################################################
sag_id = sys.argv[1]
minhash_recruits = sys.argv[2]
nmf_dat = sys.argv[3]
cov_dat = sys.argv[4]
nmf_output = sys.argv[5]
best_output = sys.argv[6]
src2sag_file = sys.argv[7]
nthreads = int(sys.argv[8])

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

minhash_df = pd.read_csv(minhash_recruits, sep='\t', header=0)
minhash_filter_df = minhash_df.loc[minhash_df['jacc_sim_avg'] >= 0.16]
sag_mh_df = minhash_filter_df.loc[minhash_filter_df['sag_id'] == sag_id]
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
            arg_list.append([sag_id, sag_mh_df, nmf_dat, cov_dat,
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
