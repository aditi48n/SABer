#!/usr/bin/env python

import sys

import hdbscan
import pandas as pd
from tqdm import tqdm

tetra_dat = sys.argv[1]
cov_dat = sys.argv[2]

# load nmf file
print('Loading Tetra and Abundance...')
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

print('Performing De Novo Clustering...')
clusterer = hdbscan.HDBSCAN(min_cluster_size=100, cluster_selection_method='eom',
                            prediction_data=True, cluster_selection_epsilon=0,
                            min_samples=100
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

cluster_df.to_csv('/home/ryan/Desktop/test_NMF/minhash_features/'
                  'CAMI_high_GoldStandardAssembly.hdbscan.tsv',
                  sep='\t', index=False
                  )

print('Denoising/cleaning Clusters...')
ns_ratio_list = []
for contig in tqdm(list(cluster_df['contig_id'].unique())):
    sub_df = cluster_df.query('contig_id == @contig')
    noise_cnt = sub_df.query('label == -1').shape[0]
    signal_cnt = sub_df.query('label != -1').shape[0]
    ns_ratio = (noise_cnt / (noise_cnt + signal_cnt)) * 100
    prob_df = sub_df.groupby(['label'])[['probabilities']].max().reset_index()
    best_ind = prob_df['probabilities'].argmax()
    best_label = prob_df['label'].iloc[best_ind]
    best_prob = prob_df['probabilities'].iloc[best_ind]
    ns_ratio_list.append([contig, best_label])

ns_ratio_df = pd.DataFrame(ns_ratio_list, columns=['contig_id', 'best_label'])
cluster_ns_df = cluster_df.merge(ns_ratio_df, on='contig_id', how='left')
no_noise_df = cluster_ns_df.query('best_label != -1')  # 'best_prob >= 0.51')
noise_df = cluster_ns_df.query('best_label == -1')  # 'best_prob < 0.51')
no_noise_df.to_csv('/home/ryan/Desktop/test_NMF/minhash_features/'
                   'CAMI_high_GoldStandardAssembly.no_noise.tsv',
                   sep='\t', index=False
                   )
noise_df.to_csv('/home/ryan/Desktop/test_NMF/minhash_features/'
                'CAMI_high_GoldStandardAssembly.noise.tsv',
                sep='\t', index=False
                )

print(set(no_noise_df['label']).intersection(set(noise_df['label'])))
print('noise subcontigs:', noise_df.shape[0])
print('cluster subcontigs:', no_noise_df.shape[0])
print('noise contigs:', len(noise_df['contig_id'].unique()))
print('cluster contigs:', len(no_noise_df['contig_id'].unique()))
print('noise clusters:', len(noise_df['label'].unique()))
print('cluster clusters:', len(no_noise_df['label'].unique()))
print('noise best clusters:', len(noise_df['label'].unique()))
print('cluster best clusters:', len(no_noise_df['label'].unique()))
