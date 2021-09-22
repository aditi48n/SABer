#!/usr/bin/env python

from os.path import join as o_join
from pathlib import Path

import hdbscan
import pandas as pd
import umap
from tqdm import tqdm


def runClusterer(mg_id, clst_path, cov_file, tetra_file, minhash_dict,
                 nthreads
                 ):
    # Convert CovM to UMAP feature table
    cov_emb = Path(o_join(clst_path, mg_id + '.covm_emb.tsv'))
    if not cov_emb.is_file():
        print('Building UMAP embedding for Coverage...')
        cov_df = pd.read_csv(cov_file, header=0, sep='\t', index_col='contigName')
        clusterable_embedding = umap.UMAP(n_neighbors=10, min_dist=0.0,
                                          n_components=int(cov_df.shape[1]),
                                          random_state=42, metric='manhattan'
                                          ).fit_transform(cov_df)
        umap_feat_df = pd.DataFrame(clusterable_embedding, index=cov_df.index.values)
        umap_feat_df.reset_index(inplace=True)
        umap_feat_df.rename(columns={'index': 'subcontig_id'}, inplace=True)
        umap_feat_df.to_csv(cov_emb, sep='\t', index=False)
    # Convert Tetra to UMAP feature table
    tetra_emb = Path(o_join(clst_path, mg_id + '.tetra_emb.tsv'))
    if not tetra_emb.is_file():
        print('Building UMAP embedding for Tetra Hz...')
        tetra_df = pd.read_csv(tetra_file, header=0, sep='\t', index_col='contig_id')
        clusterable_embedding = umap.UMAP(n_neighbors=10, min_dist=0.0, n_components=40,
                                          random_state=42, metric='manhattan'
                                          ).fit_transform(tetra_df)
        umap_feat_df = pd.DataFrame(clusterable_embedding, index=tetra_df.index.values)
        umap_feat_df.reset_index(inplace=True)
        umap_feat_df.rename(columns={'index': 'subcontig_id'}, inplace=True)
        umap_feat_df.to_csv(tetra_emb, sep='\t', index=False)

    denovo_out_file = Path(o_join(clst_path, mg_id + '.denovo_clusters.tsv'))
    noise_out_file = Path(o_join(clst_path, mg_id + '.noise.tsv'))
    if not denovo_out_file.is_file():
        print('Loading Tetra and Coverage Embeddings...')
        tetra_feat_df = pd.read_csv(tetra_emb, sep='\t', header=0, index_col='subcontig_id')
        tetra_feat_df['contig_id'] = [x.rsplit('_', 1)[0] for x in tetra_feat_df.index.values]
        # load covm file
        cov_feat_df = pd.read_csv(cov_emb, sep='\t', header=0)
        cov_feat_df.rename(columns={'contigName': 'subcontig_id'}, inplace=True)
        cov_feat_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cov_feat_df['subcontig_id']]
        cov_feat_df.set_index('subcontig_id', inplace=True)
        merge_df = tetra_feat_df.join(cov_feat_df, lsuffix='_tetra', rsuffix='_cov')
        merge_df.drop(columns=['contig_id_tetra', 'contig_id_cov'], inplace=True)
        tetra_feat_df.drop(columns=['contig_id'], inplace=True)
        cov_feat_df.drop(columns=['contig_id'], inplace=True)

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

        cluster_df.to_csv(o_join(clst_path, mg_id + '.hdbscan.tsv'),
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
        no_noise_df.to_csv(denovo_out_file, sep='\t', index=False)
        noise_df.to_csv(noise_out_file, sep='\t', index=False)
    else:
        print('Loading De Novo Clusters...')
        no_noise_df = pd.read_csv(denovo_out_file, header=0, sep='\t')
        noise_df = pd.read_csv(noise_out_file, header=0, sep='\t')

    trust_out_file = Path(o_join(clst_path, mg_id + '.trusted_clusters.tsv'))
    if minhash_dict and not trust_out_file.is_file():
        # Group clustered and noise contigs by trusted contigs
        print('Re-grouping with Trusted Contigs...')
        mh_trusted_df = minhash_dict[201]
        mh_trusted_df.rename(columns={'q_contig_id': 'contig_id'}, inplace=True)
        trust_recruit_list = []
        label_max_list = []
        contig_max_list = []
        for sag_id in tqdm(mh_trusted_df['sag_id'].unique()):
            # Subset trusted contigs by SAG
            sub_trusted_df = mh_trusted_df.query('sag_id == @sag_id and jacc_sim >= 1.0')
            trusted_contigs_list = sub_trusted_df['contig_id'].unique()
            # Gather trusted contigs subset from cluster and noise DFs
            sub_denovo_df = no_noise_df.query('contig_id in @trusted_contigs_list and '
                                              'probabilities >= 1.0'
                                              )
            sub_noise_df = noise_df.query('contig_id in @trusted_contigs_list')
            # Gather all contigs associated with trusted clusters
            trust_clust_list = list(sub_denovo_df['best_label'].unique())
            sub_label_df = no_noise_df.query('best_label in @trust_clust_list')
            # Get average max jaccard for each cluster
            label_mh_df = sub_label_df.merge(sub_trusted_df, on='contig_id', how='left')
            label_max_df = label_mh_df.groupby(['best_label'])[['jacc_sim']].mean().reset_index()
            label_max_df['sag_id'] = sag_id
            label_max_list.append(label_max_df)
            # Get max jaccard for noise labeled contigs
            if sub_noise_df.shape[0] != 0:
                noise_mh_df = sub_noise_df.merge(sub_trusted_df, on='contig_id', how='left')
                noise_max_df = noise_mh_df.groupby(['contig_id'])[['jacc_sim']].mean().reset_index()
                noise_max_df['sag_id'] = sag_id
                contig_max_list.append(noise_max_df)

        sag_label_df = pd.concat(label_max_list)
        sag_contig_df = pd.concat(contig_max_list)
        sag_label_best_df = sag_label_df.sort_values(by='jacc_sim', ascending=False
                                                     ).drop_duplicates(subset='best_label')
        sag_contig_best_df = sag_contig_df.sort_values(by='jacc_sim', ascending=False
                                                       ).drop_duplicates(subset='contig_id')
        sag_label_list = []
        for index, row in sag_label_best_df.iterrows():
            best_label = row['best_label']
            jacc_sim = row['jacc_sim']
            sub_sag_label_df = sag_label_df.query('best_label == @best_label and '
                                                  'jacc_sim == @jacc_sim'
                                                  )
            sag_label_list.append(sub_sag_label_df)

        sag_contig_list = []
        for index, row in sag_contig_best_df.iterrows():
            sag_id = row['sag_id']
            jacc_sim = row['jacc_sim']
            sub_sag_contig_df = sag_contig_df.query('sag_id == @sag_id and '
                                                    'jacc_sim == @jacc_sim'
                                                    )
            sag_contig_list.append(sub_sag_contig_df)

        label_pruned_df = pd.concat(sag_label_list)
        contig_pruned_df = pd.concat(sag_contig_list)

        sag_denovo_df = label_pruned_df.merge(no_noise_df, on='best_label', how='left')
        sag_noise_df = contig_pruned_df.merge(noise_df, on='contig_id', how='left')

        print('Building Trusted Clusters...')
        for sag_id in tqdm(mh_trusted_df['sag_id'].unique()):
            trust_cols = ['sag_id', 'contig_id']
            sub_trusted_df = mh_trusted_df.query('sag_id == @sag_id and jacc_sim >= 1.0'
                                                 )[trust_cols]
            sub_denovo_df = sag_denovo_df.query('sag_id == @sag_id')[trust_cols]
            sub_noise_df = sag_noise_df.query('sag_id == @sag_id')[trust_cols]
            trust_cat_df = pd.concat([sub_trusted_df, sub_denovo_df, sub_noise_df]
                                     ).drop_duplicates()
            trust_recruit_list.append(trust_cat_df)

        trust_recruit_df = pd.concat(trust_recruit_list)
        trust_recruit_df.rename(columns={'sag_id': 'best_label'}, inplace=True)

        trust_recruit_df.to_csv(trust_out_file, sep='\t', index=False)
    elif trust_out_file.is_file():
        print('Trust Clusters already exist...')
        trust_recruit_df = pd.read_csv(trust_out_file, sep='\t', header=0)
    else:
        print('No Trusted Contigs Provided...')
        trust_recruit_df = False

    return no_noise_df, trust_recruit_df
