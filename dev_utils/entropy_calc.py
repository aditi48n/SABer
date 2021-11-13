import glob
import hashlib
import os

import dit
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from dit.other import renyi_entropy
from scipy import sparse
from sklearn.metrics.pairwise import euclidean_distances


def calc_ref_entropy(sample_list, ref_dir, rerun_ref):
    entropy_file = os.path.join(ref_dir, 'entropy_table.tsv')
    if rerun_ref:
        # Calculate entropy for all references
        entropy_list = []
        for samp_file in sample_list:
            samp_id = samp_file.split('/')[-1].rsplit('.', 1)[0]
            if 'entropy' not in samp_id and 'table' not in samp_id:
                if samp_id.rsplit('_', 1)[1][0].isdigit():
                    samp_label = samp_id.rsplit('_', 1)[0]
                    samp_rep = samp_id.rsplit('_', 1)[1]
                else:
                    samp_label = samp_id
                    samp_rep = 0
                cov_df = pd.read_csv(samp_file, sep='\t', header=0)
                cov_df['hash_id'] = [hashlib.sha256(x.encode(encoding='utf-8')).hexdigest()
                                     for x in cov_df['contigName']
                                     ]
                depth_sum = cov_df['totalAvgDepth'].sum()
                hash_list = cov_df['hash_id'].tolist()
                relative_depth = [x / depth_sum for x in cov_df['totalAvgDepth']]
                cov_dist = dit.Distribution(hash_list, relative_depth)
                q_list = [0, 1, 2, 4, 8, 16, 32, np.inf]
                for q in q_list:
                    r_ent = renyi_entropy(cov_dist, q)
                    print(samp_id, q, r_ent)
                    entropy_list.append([samp_id, samp_label, samp_rep, q, r_ent])
                '''
                c = 0
                for col in cov_df.columns:
                    if col.split('.')[-1] == 'bam':
                        samp_id_c = samp_id # + '_' + str(c)
                        depth_sum = cov_df[col].sum()
                        hash_list = cov_df['hash_id'].tolist()
                        relative_depth = [x/depth_sum for x in cov_df[col]]
                        cov_dist = dit.Distribution(hash_list, relative_depth)
                        q_list = [0, 1, 2, 4, 8, 16, 32, np.inf]
                        for q in q_list:
                            r_ent = renyi_entropy(cov_dist, q)
                            print(samp_id_c, q, r_ent)
                            entropy_list.append([samp_id_c, samp_label, samp_rep, q, r_ent])
                        c += 1
                '''
        ent_df = pd.DataFrame(entropy_list, columns=['sample_id', 'sample_type',
                                                     'sample_rep', 'alpha',
                                                     'Renyi_Entropy'
                                                     ])
        # Have to replace the np.inf with a real value for plotting
        ent_df['alpha_int'] = ent_df['alpha'].copy()
        ent_df['alpha_int'].replace(4, 3, inplace=True)
        ent_df['alpha_int'].replace(8, 4, inplace=True)
        ent_df['alpha_int'].replace(16, 5, inplace=True)
        ent_df['alpha_int'].replace(32, 6, inplace=True)
        ent_df['alpha_int'].replace(np.inf, 7, inplace=True)
        x_labels = {0: 'Richness (a=0)', 1: 'Shannon (a=1)',
                    2: 'Simpson (a=2)', 3: '4', 4: '8', 5: '16',
                    6: '32', 7: 'Berger–Parker (a=inf)'
                    }
        ent_df['x_labels'] = [x_labels[x] for x in ent_df['alpha_int']]
        ent_df.to_csv(entropy_file, sep='\t', index=False)
    else:
        ent_df = pd.read_csv(entropy_file, sep='\t', header=0)

    return ent_df


def entropy_cluster(working_dir, ent_df):
    cluster_table = os.path.join(working_dir, 'cluster_table.tsv')
    samp2type = {x: y for x, y in zip(ent_df['sample_id'], ent_df['sample_type'])}
    piv_df = ent_df.pivot(index='sample_id', columns='alpha', values='Renyi_Entropy')
    umap_fit = umap.UMAP(n_neighbors=2, min_dist=0.0, n_components=2,
                         random_state=42
                         ).fit(piv_df)
    umap_emb = umap_fit.transform(piv_df)
    umap_df = pd.DataFrame(umap_emb, index=piv_df.index.values)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, allow_single_cluster=False,
                                prediction_data=True
                                ).fit(umap_df)

    cluster_labels = clusterer.labels_
    cluster_probs = clusterer.probabilities_
    cluster_outlier = clusterer.outlier_scores_

    umap_df['sample_type'] = [samp2type[x] for x in umap_df.index.values]
    umap_df['sample_id'] = umap_df.index.values
    umap_df['cluster'] = cluster_labels
    umap_df['probabilities'] = cluster_probs
    umap_df['outlier_scores'] = cluster_outlier

    ent_umap_df = ent_df.merge(umap_df, on=['sample_id', 'sample_type'], how='left')
    ent_best_df = find_best_match(piv_df, ent_umap_df)
    ent_best_df.to_csv(cluster_table, sep='\t', index=False)

    return umap_df, ent_best_df, piv_df, umap_fit, clusterer


def find_best_match(piv_df, ent_umap_df):
    # Closest ref sample methods
    outlier_list = list(ent_umap_df.query("cluster == -1")['sample_id'])
    cmpr_list = []
    for r1, row1 in piv_df.iterrows():
        keep_diff = [r1, '', np.inf]
        for r2, row2 in piv_df.iterrows():
            diff = ((row1 - row2).abs()).sum()
            if diff < keep_diff[2] and r1 != r2 and r2 not in outlier_list:
                keep_diff = [r1, r2, diff]
        cmpr_list.append(keep_diff)
    cmpr_df = pd.DataFrame(cmpr_list, columns=['sample_id', 'best_match', 'diff'])
    ent_best_df = ent_umap_df.merge(cmpr_df, on='sample_id', how='left')

    return ent_best_df


def real_best_match(piv_df, real_piv_df, real_umap_df, working_dir):
    # Closest ref sample methods
    r_cmpr_list = []
    for r1, row1 in real_piv_df.iterrows():
        keep_diff = [r1, '', np.inf]
        for r2, row2 in piv_df.iterrows():
            diff = ((row1 - row2).abs()).sum()
            if diff < keep_diff[2] and r1 != r2:
                keep_diff = [r1, r2, diff]
        r_cmpr_list.append(keep_diff)
    r_cmpr_df = pd.DataFrame(r_cmpr_list, columns=['sample_id', 'best_match', 'diff'])
    best_df = real_umap_df.merge(r_cmpr_df, on='sample_id', how='left')
    best_df.to_csv(os.path.join(working_dir, 'cluster_table.tsv'), sep='\t', index=False)
    return best_df


def real_cluster(clusterer, real_df, umap_df, umap_fit):
    # Assign real data to clusters
    real_piv_df = real_df.pivot(index='sample_id', columns='alpha', values='Renyi_Entropy')
    umap_emb = umap_fit.transform(real_piv_df)
    umap_df = pd.DataFrame(umap_emb, index=real_piv_df.index.values)
    test_labels, strengths = hdbscan.approximate_predict(clusterer, umap_df)
    cluster_labels = test_labels
    cluster_probs = strengths
    samp2type = {x: y for x, y in zip(real_df['sample_id'], real_df['sample_type'])}
    umap_df['sample_type'] = [samp2type[x] for x in umap_df.index.values]
    umap_df['sample_id'] = umap_df.index.values
    umap_df['cluster'] = cluster_labels
    umap_df['probabilities'] = cluster_probs
    return real_piv_df, umap_df


def calc_real_entrophy(mba_cov_list, working_dir):
    entropy_list = []
    for samp_file in mba_cov_list:
        samp_id = samp_file.split('/')[-1].rsplit('.', 1)[0]
        print(samp_id)
        if samp_id.rsplit('_', 1)[1][0].isdigit():
            samp_label = samp_id.rsplit('_', 1)[0]
            samp_rep = samp_id.rsplit('_', 1)[1]
        else:
            samp_label = samp_id
            samp_rep = 0
        cov_df = pd.read_csv(samp_file, sep='\t', header=0)
        cov_df['hash_id'] = [hashlib.sha256(x.encode(encoding='utf-8')).hexdigest()
                             for x in cov_df['contigName']
                             ]
        depth_sum = cov_df['totalAvgDepth'].sum()
        cov_df['relative_depth'] = [x / depth_sum for x in cov_df['totalAvgDepth']]
        cov_dist = dit.Distribution(cov_df['hash_id'].tolist(),
                                    cov_df['relative_depth'].tolist()
                                    )
        q_list = [0, 1, 2, 4, 8, 16, 32, np.inf]
        for q in q_list:
            r_ent = renyi_entropy(cov_dist, q)
            print(q, r_ent)
            entropy_list.append([samp_id, samp_label, samp_rep, q, r_ent])
    real_df = pd.DataFrame(entropy_list, columns=['sample_id', 'sample_type',
                                                  'sample_rep', 'alpha',
                                                  'Renyi_Entropy'
                                                  ])
    # Have to replace the np.inf with a real value for plotting
    real_df['alpha_int'] = real_df['alpha'].copy()
    real_df['alpha_int'].replace(4, 3, inplace=True)
    real_df['alpha_int'].replace(8, 4, inplace=True)
    real_df['alpha_int'].replace(16, 5, inplace=True)
    real_df['alpha_int'].replace(32, 6, inplace=True)
    real_df['alpha_int'].replace(np.inf, 7, inplace=True)
    x_labels = {0: 'Richness (a=0)', 1: 'Shannon (a=1)',
                2: 'Simpson (a=2)', 3: '4', 4: '8', 5: '16',
                6: '32', 7: 'Berger–Parker (a=inf)'
                }
    real_df['x_labels'] = [x_labels[x] for x in real_df['alpha_int']]
    real_df.to_csv(os.path.join(working_dir, 'entropy_table.tsv'), sep='\t', index=False)

    return real_df


def remove_outliers(ent_best_df, real_merge_df):
    # If labeled as an outlier, take the closest match
    keep_cols = ['sample_id', 'sample_type', 'alpha', 'Renyi_Entropy', 'alpha_int',
                 'x_labels', 0, 1, 'cluster', 'probabilities', 'best_match', 'diff'
                 ]
    best_merge_df = pd.concat([ent_best_df[keep_cols], real_merge_df[keep_cols]])
    samp2clust = dict(zip(best_merge_df['sample_id'], best_merge_df['cluster']))
    best_merge_df['cluster'] = [c if c != -1 else samp2clust[s] for s, c in
                                zip(best_merge_df['best_match'], best_merge_df['cluster'])
                                ]
    best_merge_df['cluster'] = [c if c != -1 else samp2clust[s] for s, c in
                                zip(best_merge_df['best_match'], best_merge_df['cluster'])
                                ]

    return best_merge_df


def plot_renyi_entropy(working_dir, ent_df, read_type):
    # Build all the reference plots and run clustering
    type_list = ent_df['sample_type'].unique()
    cpal = {x: y for x, y in zip(type_list, sns.color_palette(n_colors=len(type_list)))}
    cpal[read_type] = 'black'
    sns.set(rc={'figure.figsize': (12, 8)})
    sns.set_style("white")
    p = sns.catplot(x="x_labels", y="Renyi_Entropy", hue="sample_type", col="sample_type",
                    data=ent_df, palette=cpal, col_wrap=3, legend_out=True
                    )
    p.map_dataframe(sns.boxplot, x="x_labels", y="Renyi_Entropy",
                    data=ent_df, boxprops={'facecolor': 'None'},
                    whiskerprops={'linewidth': 1},
                    showfliers=False
                    )
    p.set_xticklabels(rotation=45)
    lgd_dat = p._legend_data
    p.add_legend(legend_data={x: lgd_dat[x] for x in lgd_dat.keys() if x in cpal.keys()})
    p.savefig(os.path.join(working_dir, 'entropy_plot.png'), bbox_inches='tight')
    plt.clf()
    plt.close()

    return cpal


def plot_euc_dist(working_dir, ent_df, cpal):
    # Calculate euclidean distances and plot
    samp2type = {x: y for x, y in zip(ent_df['sample_id'], ent_df['sample_type'])}
    piv_df = ent_df.pivot(index='sample_id', columns='alpha', values='Renyi_Entropy')
    A_sparse = sparse.csr_matrix(piv_df)
    similarities = euclidean_distances(A_sparse)
    euc_df = pd.DataFrame(similarities, index=piv_df.index.values, columns=piv_df.index.values)
    euc_unstack_df = euc_df.unstack().reset_index()
    euc_unstack_df.columns = ['sample1', 'sample2', 'euclidean_distances']
    euc_unstack_df['samp1_label'] = [samp2type[x] for x in euc_unstack_df['sample1']]
    euc_unstack_df['samp2_label'] = [samp2type[x] for x in euc_unstack_df['sample2']]
    sns.set(rc={'figure.figsize': (12, 8)})
    sns.set_style("white")
    b = sns.catplot(x="samp1_label", y="euclidean_distances", hue="samp2_label", col="samp2_label",
                    data=euc_unstack_df, palette=cpal, col_wrap=3, legend_out=True
                    )
    b.map_dataframe(sns.boxplot, x="samp1_label", y="euclidean_distances",
                    data=euc_unstack_df, boxprops={'facecolor': 'None'},
                    whiskerprops={'linewidth': 1},
                    showfliers=False
                    )
    b.set_xticklabels(rotation=45)
    lgd_dat = b._legend_data
    b.add_legend(legend_data={x: lgd_dat[x] for x in lgd_dat.keys() if x in cpal.keys()})
    b.savefig(os.path.join(working_dir, 'similarity_plot.png'), bbox_inches='tight')
    plt.clf()
    plt.close()


def plot_ent_clust(working_dir, ent_umap_df, cpal):
    sns.set(rc={'figure.figsize': (12, 8)})
    sns.set_style("white")
    mark_list = ['.', 'v', '^', '<', '>', '8', 's', 'p', 'P', '*', 'H', 'X', 'D', 'd', 'o']
    b = sns.scatterplot(x=0, y=1, hue="sample_type", style='cluster',
                        data=ent_umap_df, palette=cpal, markers=mark_list, s=200
                        )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig = b.get_figure()
    fig.savefig(os.path.join(working_dir, 'entropy_clusters.png'), bbox_inches='tight')
    plt.clf()
    plt.close()

    sns.set(rc={'figure.figsize': (12, 8)})
    sns.set_style("white")
    p = sns.catplot(x="x_labels", y="Renyi_Entropy", hue="sample_type",
                    col="cluster", data=ent_umap_df, col_wrap=4, palette=cpal,
                    legend=False
                    )
    p.map_dataframe(sns.boxplot, x="x_labels", y="Renyi_Entropy",
                    data=ent_umap_df, boxprops={'facecolor': 'None'},
                    whiskerprops={'linewidth': 1},
                    showfliers=False
                    )
    p.set_xticklabels(rotation=45)
    lgd_dat = p._legend_data
    p.add_legend(legend_data={x: lgd_dat[x] for x in lgd_dat.keys() if x in cpal.keys()})
    p.savefig(os.path.join(working_dir, 'clustent_plot.png'), bbox_inches='tight')
    plt.clf()
    plt.close()


def calc_real_entropy(working_dir, mba_cov_list):
    rerun_ref = False  # to re-calc reference profiles, set to True
    make_plots = True  # if you want plots, set to True

    # Calculate entropy for all references
    ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           'renyi_entropy/references/'
                           )
    sample_list = glob.glob(os.path.join(ref_dir, "*.tsv"))
    ent_df = calc_ref_entropy(sample_list, ref_dir, rerun_ref)
    ent_results = entropy_cluster(ref_dir, ent_df)
    umap_df = ent_results[0]
    ent_best_df = ent_results[1]
    piv_df = ent_results[2]
    umap_fit = ent_results[3]
    clusterer = ent_results[4]

    # Cluster real samples
    real_df = calc_real_entrophy(mba_cov_list, working_dir)
    real_piv_df, real_umap_df = real_cluster(clusterer, real_df, umap_df, umap_fit)
    real_best_df = real_best_match(piv_df, real_piv_df, real_umap_df, working_dir)
    real_merge_df = real_df.merge(real_best_df, on=['sample_id', 'sample_type'], how='left')

    # Replace outliers with best match
    best_merge_df = remove_outliers(ent_best_df, real_merge_df)

    # Build plots
    real_type = real_df['sample_type'].values[0]
    if make_plots == True:
        cpal = plot_renyi_entropy(ref_dir, best_merge_df, real_type)
        plot_euc_dist(ref_dir, best_merge_df, cpal)
        plot_ent_clust(ref_dir, best_merge_df, cpal)

    return real_merge_df


###############################################################################################
###############################################################################################
working_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           'renyi_entropy/SI/'
                           )
mba_cov_list = glob.glob(os.path.join(working_dir, "SI*.tsv"))
best_params_df = calc_real_entropy(working_dir, mba_cov_list)
