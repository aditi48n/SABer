import matplotlib
import pandas as pd

matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 2000000

import matplotlib.pyplot as plt
import seaborn as sns;
from tqdm import tqdm

sns.set(style="ticks", color_codes=True)
pd.set_option('display.max_columns', None)
pd.set_option('mode.chained_assignment', None)
sns.set_context("poster")
sns.set_style('whitegrid')
sns.set(font_scale=1.0)

# TODO: build plot to investigate best hyperparams based on starting completeness
pred_file = '/home/ryan/Desktop/test_NMF/CAMI_high_GoldStandardAssembly.all_scores.tsv'
src2contig_file = '/home/ryan/Desktop/test_NMF/src2contig_map.tsv'
abund_file = '/home/ryan/Desktop/test_NMF/CAMI_high_GoldStandardAssembly.nmf_trans_20.tsv'
sag2cami_file = '/home/ryan/Desktop/test_NMF/sag2cami_map.tsv'

pred_df = pd.read_csv(pred_file, header=0, sep='\t')
src2contig_df = pd.read_csv(src2contig_file, header=0, sep='\t')
abund_df = pd.read_csv(abund_file, header=0, sep='\t')
sag2cami_df = pd.read_csv(sag2cami_file, header=0, sep='\t')

count_df = abund_df.groupby(['contig_id'])['subcontig_id'].count().reset_index()

src2contig_df = src2contig_df[src2contig_df['CAMI_genomeID'].notna()]
src2contig_df = pd.merge(src2contig_df, count_df, left_on='@@SEQUENCEID', right_on='contig_id', how='left')
src_count_df = src2contig_df.groupby(['CAMI_genomeID'])['subcontig_id'].sum().reset_index()
src_count_df.columns = ['CAMI_genomeID', 'subcontig_count']
pred_df = pd.merge(pred_df, sag2cami_df, on='sag_id', how='left')
pred_df = pd.merge(pred_df, src_count_df, on='CAMI_genomeID', how='left')
pred_df = pred_df.dropna()
pred_df = pred_df.loc[((pred_df['TP'] > 0) | (pred_df['FP'] > 0))]
pred_df['nu_gamma'] = [str(x[0]) + '_' + str(x[1]) for x in
                       zip(pred_df['nu'], pred_df['gamma'])
                       ]
print(pred_df.head())
print(pred_df.shape)
incl_dict = {'majority': 0, 'all': 1}
lev_dict = {'strain': 1, 'exact': 0}
gamma_dict = {'scale': 13, '1e-06': 1, '1e-05': 2, '0.0001': 3, '0.001': 4, '0.01': 5, '0.1': 6,
              '1': 7, '10': 8, '100': 9, '1000': 10, '10000': 11, '100000': 12
              }
'''
for level in lev_dict:
    lev_df = pred_df.loc[pred_df['level'] == level]
    for inclusion in incl_dict:
        incl_df = lev_df.loc[lev_df['inclusion'] == inclusion]
        for nu in incl_df['nu'].unique():
            print(level, inclusion, nu)
            nu_df = incl_df.loc[incl_df['nu'] == nu]
            mean_df = nu_df.groupby(['gamma'])['sensitivity', 'precision'].mean().reset_index()
            g = sns.relplot(data=mean_df, x='sensitivity', y='precision', col='gamma', col_wrap=4,
                            kind='line'
                            )
            plt.savefig('PR_plots/PRs/' + level + '_' + inclusion + '_' + str(nu) + '_PR_curve.png')
            plt.clf()
            plt.close()

sys.exit()
'''

pred_df['round1_sensitivity'] = pred_df['sensitivity'].round(1)
pred_df['round1_precision'] = pred_df['precision'].round(1)
pred_df['round1_MCC'] = pred_df['MCC'].round(1)
pred_df['round2_sensitivity'] = pred_df['sensitivity'].round(2)
pred_df['round2_precision'] = pred_df['precision'].round(2)
pred_df['round2_MCC'] = pred_df['MCC'].round(2)

flierprops = dict(markerfacecolor='0.75', markersize=5, markeredgecolor='w',
                  linestyle='none')

'''
pred_df = pred_df.sort_values(['round1_sensitivity', 'round1_precision'], ascending=[False, False])
PR_df = pred_df.drop_duplicates(subset=['sag_id'], keep='first')
PR_df = PR_df[['sag_id', 'round1_sensitivity', 'round1_precision']]
PR_df.columns = ['sag_id', 'best_sensitivity', 'best_precision']
merge_df = pd.merge(pred_df, PR_df, on=['sag_id'])
filter_df = merge_df.loc[((merge_df['round1_sensitivity'] >= merge_df['best_sensitivity']) &
                          (merge_df['round1_precision'] >= merge_df['best_precision'])
                          )]

filter_df['inclusion_sorter'] = [incl_dict[x] for x in filter_df['inclusion']]
filter_df['level_sorter'] = [lev_dict[x] for x in filter_df['level']]
filter_df['gamma_sorter'] = [gamma_dict[x] for x in filter_df['gamma']]
filter_df = filter_df.sort_values(['nu', 'gamma_sorter', 'inclusion_sorter', 'level_sorter'],
                                  ascending=[False, True, True, True]
                                  )
num_1_df = filter_df.drop_duplicates(subset=['sag_id'], keep='first')

keep_list = ['sag_id', 'level', 'inclusion', 'gamma', 'nu', 'precision', 'MCC', 'sensitivity']
pred_stack_df = num_1_df[keep_list].set_index(['sag_id', 'level', 'inclusion',
                                               'gamma', 'nu']).stack().reset_index()
pred_stack_df.columns = ['sag_id', 'level', 'inclusion',
                         'gamma', 'nu', 'metric', 'score'
                         ]

pred_stack_df['combo'] = [x[0] + '_' + x[1] for x in zip(pred_stack_df['level'],
                                                         pred_stack_df['inclusion'])
                          ]

mean_df = pred_stack_df.groupby(['level', 'inclusion', 'metric']
                                )['score'].mean().unstack('metric').reset_index()
mean_df['round2_sensitivity'] = mean_df['sensitivity'].round(2)
mean_df['round2_precision'] = mean_df['precision'].round(2)
mean_df['round2_MCC'] = mean_df['MCC'].round(2)

mean_df = mean_df.sort_values(['round2_sensitivity', 'round2_precision'], ascending=[False, False])
mean_df.to_csv('PR_plots/Mean_stats.tsv', sep='\t', index=False)

ax = sns.catplot(x="metric", y="score", hue="level", kind='box',
                 data=pred_stack_df, aspect=2, palette=sns.light_palette("black"),
                 flierprops=flierprops)
plt.savefig('PR_plots/level_boxplot.png', bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

ax = sns.catplot(x="metric", y="score", hue="inclusion", kind='box',
                 data=pred_stack_df, aspect=2, palette=sns.light_palette("black"),
                 flierprops=flierprops)
plt.savefig('PR_plots/inclusion_boxplot.png', bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

ax = sns.catplot(x="metric", y="score", hue="combo", kind='box',
                 data=pred_stack_df, aspect=2, palette=sns.light_palette("black"),
                 flierprops=flierprops)
plt.savefig('PR_plots/combo_boxplot.png', bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

count_df = num_1_df[['level', 'inclusion', 'sag_id']
].groupby(['level', 'inclusion']
          ).count().reset_index()
count_df.columns = ['level', 'inclusion', 'count']
count_df['combo'] = [x[0] + '_' + x[1] for x in zip(count_df['level'],
                                                    count_df['inclusion'])
                     ]

count_df = count_df.sort_values(['count'], ascending=[False])
g = sns.barplot(data=count_df, x="count", y="combo")
plt.savefig('PR_plots/combo_barplot.png', bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

top_level = count_df['level'].iloc[0]
top_inclusion = count_df['inclusion'].iloc[0]
top_df = pred_df.loc[((pred_df['level'] == top_level) &
                      (pred_df['inclusion'] == top_inclusion))
]
'''
best_sub_list = []
for sag_id in tqdm(set(pred_df['sag_id'])):
    # print(sag_id)
    sag_top_df = pred_df.loc[pred_df['sag_id'] == sag_id]
    sag_top_df['rank1_sensitivity'] = (sag_top_df['round1_sensitivity']).astype(float).rank(
        method='dense', ascending=False).astype(float)
    sag_top_df['rank1_precision'] = (sag_top_df['round1_precision']).astype(float).rank(
        method='dense', ascending=False).astype(float)
    sag_top_df['rank1_MCC'] = (sag_top_df['round1_MCC']).astype(float).rank(
        method='dense', ascending=False).astype(float)
    sag_top_df['rank2_sensitivity'] = (sag_top_df['round2_sensitivity']).astype(float).rank(
        method='dense', ascending=False).astype(float)
    sag_top_df['rank2_precision'] = (sag_top_df['round2_precision']).astype(float).rank(
        method='dense', ascending=False).astype(float)
    sag_top_df['rank2_MCC'] = (sag_top_df['round2_MCC']).astype(float).rank(
        method='dense', ascending=False).astype(float)

    rank_df = sag_top_df[
        ['sag_id', 'level', 'inclusion', 'nu_gamma', 'nu', 'gamma', 'rank1_MCC', 'rank1_sensitivity',
         'rank1_precision', 'rank2_MCC', 'rank2_sensitivity', 'rank2_precision', 'precision',
         'MCC', 'sensitivity'
         ]]
    rank_df = rank_df.sort_values(['rank1_precision', 'rank1_sensitivity',
                                   'rank2_precision', 'rank2_sensitivity'
                                   ], ascending=[True, True, True, True])
    '''
    rank_df = rank_df.sort_values(['rank1_MCC', 'rank1_sensitivity', 'rank1_precision', 'rank2_MCC',
                                   'rank2_precision', 'rank2_sensitivity'
                                   ], ascending=[True, True, True, True, True, True])
    '''
    top_ranks = rank_df[['rank1_MCC', 'rank1_sensitivity', 'rank1_precision', 'rank2_MCC',
                         'rank2_precision', 'rank2_sensitivity']].iloc[0]
    sub_rank_df = rank_df.loc[((rank_df['rank1_MCC'] == top_ranks['rank1_MCC']) &
                               (rank_df['rank1_sensitivity'] == top_ranks['rank1_sensitivity']) &
                               (rank_df['rank1_precision'] == top_ranks['rank1_precision']) &
                               (rank_df['rank2_MCC'] == top_ranks['rank2_MCC']) &
                               (rank_df['rank2_precision'] == top_ranks['rank2_precision']) &
                               (rank_df['rank2_sensitivity'] == top_ranks['rank2_sensitivity'])
                               )]

    best_sub_list.append(sub_rank_df)

concat_df = pd.concat(best_sub_list)
# select the config that overfits the least
concat_df['gamma_sorter'] = [gamma_dict[x] for x in concat_df['gamma']]
concat_df = concat_df.sort_values(['rank1_precision', 'rank1_sensitivity',
                                   'rank2_precision', 'rank2_sensitivity', 'nu', 'gamma_sorter'],
                                  ascending=[True, True, True, True, False, True])
'''
concat_df = concat_df.sort_values(['rank1_MCC', 'rank1_sensitivity', 'rank1_precision', 'rank2_MCC',
                                   'rank2_precision', 'rank2_sensitivity', 'nu', 'gamma_sorter'],
                                  ascending=[True, True, True, True, True, True, False, True])
'''
sag_dedup_df = concat_df.drop_duplicates(subset='sag_id', keep='first')

keep_list = ['sag_id', 'level', 'inclusion', 'gamma', 'nu', 'precision', 'MCC', 'sensitivity']
pred_stack_df = concat_df[keep_list].set_index(['sag_id', 'level', 'inclusion',
                                                'gamma', 'nu']).stack().reset_index()
pred_stack_df.columns = ['sag_id', 'level', 'inclusion',
                         'gamma', 'nu', 'metric', 'score'
                         ]
pred_stack_df['nu_gamma'] = [str(x[0]) + '_' + str(x[1]) for x in
                             zip(pred_stack_df['nu'], pred_stack_df['gamma'])
                             ]
pred_stack_df['gamma_sorter'] = [gamma_dict[x] for x in pred_stack_df['gamma']]
pred_stack_df = pred_stack_df.sort_values(['nu', 'gamma_sorter'], ascending=[True, True])

ax = sns.catplot(x="metric", y="score", hue="gamma", kind='box',
                 data=pred_stack_df, aspect=2, palette=sns.light_palette("black"),
                 flierprops=flierprops)
plt.savefig('/home/ryan/Desktop/test_NMF/PR_plots_all/gamma_boxplot.png',
            bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

ax = sns.catplot(x="metric", y="score", hue="nu", kind='box',
                 data=pred_stack_df, aspect=2, palette=sns.light_palette("black"),
                 flierprops=flierprops)
plt.savefig('/home/ryan/Desktop/test_NMF/PR_plots_all/nu_boxplot.png',
            bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

print(concat_df.head())
top_nu_list = []
top_gamma_list = []
for level in lev_dict:
    lev_df = concat_df.loc[concat_df['level'] == level]
    for inclusion in incl_dict:
        print(level, inclusion)
        incl_df = lev_df.loc[lev_df['inclusion'] == inclusion]
        count_df = incl_df[['level', 'inclusion', 'nu', 'gamma', 'nu_gamma']
        ].groupby(['level', 'inclusion', 'nu', 'gamma']).count().reset_index()
        print(count_df.head())
        print(count_df.shape)
        count_df.columns = ['level', 'inclusion', 'nu', 'gamma', 'count']
        count_df = count_df.sort_values(['count'], ascending=[False])
        count_df.to_csv(
            '/home/ryan/Desktop/test_NMF/PR_plots_all/' + level + '_' + inclusion + '_top_ranked_params.tsv',
            index=False, sep='\t'
            )
        top_level = count_df['level'].iloc[0]
        top_inclusion = count_df['inclusion'].iloc[0]
        top_nu = count_df['nu'].iloc[0]
        top_gamma = count_df['gamma'].iloc[0]
        top_nu_list.append(top_nu)
        top_gamma_list.append(top_gamma)
        print(top_level, top_inclusion, top_nu, top_gamma)
        nu_order = list(sorted(set(count_df['nu'])))
        g = sns.FacetGrid(count_df, col="gamma", col_wrap=4)
        g.map(sns.barplot, "nu", "count", order=nu_order, ci=None)
        plt.savefig('/home/ryan/Desktop/test_NMF/PR_plots_all/' + level + '_' + inclusion + '_nu_gamma_facetplot.png')
        plt.clf()
        plt.close()

# Rank the configs per sag
top_config_df = concat_df.loc[((concat_df['nu'].isin(top_nu_list)) &
                               (concat_df['gamma'].isin(top_gamma_list))
                               )]
grp_metrics_df = top_config_df[['level', 'inclusion', 'nu_gamma', 'nu', 'gamma', 'precision',
                                'MCC', 'sensitivity'
                                ]].groupby(['level', 'inclusion', 'nu_gamma', 'nu', 'gamma']
                                           ).agg(precision_mean=('precision', 'mean'),
                                                 precision_cnt=('precision', 'count'),
                                                 sensitivity_mean=('sensitivity', 'mean'),
                                                 sensitivity_cnt=('sensitivity', 'count'),
                                                 MCC_mean=('MCC', 'mean'),
                                                 MCC_cnt=('MCC', 'count')).reset_index()

grp_metrics_df.to_csv('/home/ryan/Desktop/test_NMF/PR_plots_all/top_config_stats.tsv',
                      sep='\t', index=False)

best_metrics_df = grp_metrics_df.loc[grp_metrics_df['MCC_cnt'] == grp_metrics_df['MCC_cnt'].max()]
best_metrics_df.to_csv('/home/ryan/Desktop/test_NMF/PR_plots_all/best_config_stats.tsv',
                       sep='\t', index=False)
best_nu = best_metrics_df['nu'].iloc[0]
best_gamma = best_metrics_df['gamma'].iloc[0]

best_config_df = concat_df.loc[((concat_df['nu'] == best_nu) &
                                (concat_df['gamma'] == best_gamma))
]
mean_P = best_config_df['precision'].mean() * 100
mean_MCC = best_config_df['MCC'].mean() * 100
mean_R = best_config_df['sensitivity'].mean() * 100
max_P = best_config_df['precision'].max() * 100
max_MCC = best_config_df['MCC'].max() * 100
max_R = best_config_df['sensitivity'].max() * 100
min_P = best_config_df['precision'].min() * 100
min_MCC = best_config_df['MCC'].min() * 100
min_R = best_config_df['sensitivity'].min() * 100

g = sns.displot(best_config_df, x="rank2_precision")
text_str = ''.join(['Mean:\n  P=', str(round(mean_P, 2)), '\n  R=', str(round(mean_R, 2)),
                    '\n  MCC=', str(round(mean_MCC, 2))
                    ])
for ax in g.axes.flat:
    ax.text(7, 200, text_str, fontsize=9)
plt.savefig('/home/ryan/Desktop/test_NMF/PR_plots_all/best_config_histo_precision.png',
            bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

g = sns.displot(best_config_df, x="rank2_sensitivity")
text_str = ''.join(['Mean:\n  P=', str(round(mean_P, 2)), '\n  R=', str(round(mean_R, 2)),
                    '\n  MCC=', str(round(mean_MCC, 2))
                    ])
for ax in g.axes.flat:
    ax.text(7, 100, text_str, fontsize=9)
plt.savefig('/home/ryan/Desktop/test_NMF/PR_plots_all/best_config_histo_sensitivity.png',
            bbox_inches='tight', dpi=300)
plt.clf()
plt.close()
