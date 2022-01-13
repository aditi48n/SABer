import os
import sys
from functools import reduce

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from scipy.stats import kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# plot aestetics
sns.set_context("paper")

saber_single_file = sys.argv[1]
saber_multi_file = sys.argv[2]
unitem_single_file = sys.argv[3]
unitem_multi_file = sys.argv[4]
# vamb_multi_file = sys.argv[5] # when it's ready

# working directory
workdir = os.path.dirname(saber_single_file)

# column renaming/mapping dictionaries
type2label = {'CAMI_II_Airways': 'Air',
              'CAMI_II_Gastrointestinal': 'GI',
              'CAMI_II_Oral': 'Oral',
              'CAMI_II_Skin': 'Skin',
              'CAMI_II_Urogenital': 'Urog'
              }
algo2rank = {'denovo': 0, 'hdbscan': 1,
             'ocsvm': 2, 'intersect': 3
             }
type2rank = {'CAMI_II_Airways': 0,
             'CAMI_II_Gastrointestinal': 1,
             'CAMI_II_Oral': 2,
             'CAMI_II_Skin': 3,
             'CAMI_II_Urogenital': 4
             }
mode2rank = {'majority_rule': 0,
             'best_cluster': 1,
             'best_match': 2
             }
param2rank = {'very_relaxed': 0,
              'relaxed': 1,
              'strict': 2,
              'very_strict': 3
              }
binner2rank = {'maxbin_ms40': 0,
               'maxbin_ms107': 1,
               'metabat_specific': 2,
               'metabat_veryspecific': 3,
               'metabat_superspecific': 4,
               'metabat_sensitive': 5,
               'metabat_verysensitive': 6,
               'metabat2': 7
               }
'''
########################################################################################################################
# Single SABer - Near Complete, Absolute
########################################################################################################################
print('############################################################')
print('# Single SABer')
print('############################################################')

ss_df = pd.read_csv(saber_single_file, header=0, sep='\t')
ss_df['label'] = [type2label[x] for x in ss_df['sample_type']]
ss_df['algo_rank'] = [algo2rank[x] for x in ss_df['algo']]
ss_df['type_rank'] = [type2rank[x] for x in ss_df['sample_type']]
ss_df['mode_rank'] = [mode2rank[x] for x in ss_df['mode']]
ss_df['param_rank'] = [param2rank[x] for x in ss_df['param_set']]
ss_df['mode_paramset'] = [str(x) + '_' + str(y) for x, y in
                          zip(ss_df['mode'], ss_df['param_set'])
                          ]
ss_df['algo_param'] = ['_'.join([str(x), str(y)]) for x, y in
                       zip(ss_df['algo'], ss_df['param_set'])
                       ]
ss_df['label_sample'] = ['_'.join([str(x), str(y)]) for x, y in
                         zip(ss_df['label'], ss_df['sample_id'])
                         ]
ss_abs_str_df = ss_df.query("level == 'strain_absolute'")
ss_abs_str_df.sort_values(by=['type_rank', 'algo_rank',
                              'mode_rank', 'param_set'
                              ], inplace=True)
ss_abs_str_median_df = ss_abs_str_df.groupby(['mode', 'algo']
                                             )[['ext_nc_uniq'
                                                ]].median().reset_index()
ss_abs_str_median_df.columns = ['mode', 'algo', 'median']
ss_abs_str_mean_df = ss_abs_str_df.groupby(['mode', 'algo']
                                           )[['ext_nc_uniq'
                                              ]].mean().reset_index()
ss_abs_str_mean_df.columns = ['mode', 'algo', 'mean']
ss_abs_str_std_df = ss_abs_str_df.groupby(['mode', 'algo']
                                          )[['ext_nc_uniq'
                                             ]].std().reset_index()
ss_abs_str_std_df.columns = ['mode', 'algo', 'stdev']
stats_df_list = [ss_abs_str_median_df, ss_abs_str_mean_df, ss_abs_str_std_df]
ss_abs_str_stats_df = reduce(lambda x, y: pd.merge(x, y, on=['mode', 'algo']), stats_df_list)
ss_abs_str_stats_df.sort_values(by='mean', ascending=False, inplace=True)

print(ss_abs_str_stats_df)

MR_count = 0
BC_count = 0
BM_count = 0
for algo_param in ss_abs_str_df['algo_param'].unique():
    sub_ss_df = ss_abs_str_df.query("algo_param == @algo_param")
    mr_count = sub_ss_df.query("mode == 'majority_rule'")['ext_nc_uniq'].sum()
    bc_count = sub_ss_df.query("mode == 'best_cluster'")['ext_nc_uniq'].sum()
    bm_count = sub_ss_df.query("mode == 'best_match'")['ext_nc_uniq'].sum()
    test_df = sub_ss_df.pivot(index='label_sample', columns='mode', values='ext_nc_uniq')

    # stats f_oneway functions takes the groups as input and returns ANOVA F and p value
    fvalue, pvalue = stats.f_oneway(test_df['majority_rule'],
                                    test_df['best_cluster'],
                                    test_df['best_match']
                                    )
    m_comp = pairwise_tukeyhsd(endog=sub_ss_df['ext_nc_uniq'], groups=sub_ss_df['mode'], alpha=0.05)
    stat, p = wilcoxon(test_df['majority_rule'], test_df['best_cluster'])

    print(f"\nThe Algorithm tested is {algo_param}")
    print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
    print(f"\nResults of Tukey HSD test:")
    print(m_comp)
    print(f"\nResults of Wilcoxon Signed-Rank Test:")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')
    stat, p = kruskal(test_df['majority_rule'],
                      test_df['best_cluster'],
                      test_df['best_match']
                      )
    print(f"\nResults of Kruskal-Wallis H Test:")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
    print(f"\nThe total number of NC bins for each mode is:\n")
    print(f"\t Majority Rule: {mr_count}")
    print(f"\t Best Cluster: {bc_count}")
    print(f"\t Best Match: {bm_count}")
    MR_count += mr_count
    BC_count += bc_count
    BM_count += bm_count

print(f"Total Majority Rule: {MR_count}")
print(f"Total Best Cluster: {BC_count}")
print(f"Total Best Match: {BM_count}")
lengs = len(ss_abs_str_df['algo_param'].unique())
print(f"Average Majority Rule: {MR_count / lengs}")
print(f"Average Best Cluster: {BC_count / lengs}")
print(f"Average Best Match: {BM_count / lengs}")

# Boxplots for mode and param set
ss_box = sns.catplot(x="mode", y="ext_nc_uniq", hue="algo",
                     col="param_set", col_wrap=2,
                     kind="box", data=ss_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.single.absolute.NC.mode_param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Boxplots for mode
ss_box = sns.catplot(x="mode", y="ext_nc_uniq", hue="algo",
                     kind="box", data=ss_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.single.absolute.NC.mode.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Boxplots for param set
ss_box = sns.catplot(x="param_set", y="ext_nc_uniq", hue="algo",
                     kind="box", data=ss_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.single.absolute.NC.param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Medium Quality
ss_abs_str_median_df = ss_abs_str_df.groupby(['mode', 'algo']
                                             )[['ext_mq_uniq'
                                                ]].median().reset_index()
ss_abs_str_median_df.columns = ['mode', 'algo', 'median']
ss_abs_str_mean_df = ss_abs_str_df.groupby(['mode', 'algo']
                                           )[['ext_mq_uniq'
                                              ]].mean().reset_index()
ss_abs_str_mean_df.columns = ['mode', 'algo', 'mean']
ss_abs_str_std_df = ss_abs_str_df.groupby(['mode', 'algo']
                                          )[['ext_mq_uniq'
                                             ]].std().reset_index()
ss_abs_str_std_df.columns = ['mode', 'algo', 'stdev']
stats_df_list = [ss_abs_str_median_df, ss_abs_str_mean_df, ss_abs_str_std_df]
ss_abs_str_stats_df = reduce(lambda x, y: pd.merge(x, y, on=['mode', 'algo']), stats_df_list)
ss_abs_str_stats_df.sort_values(by='mean', ascending=False, inplace=True)

print(ss_abs_str_stats_df)

MR_count = 0
BC_count = 0
BM_count = 0
for algo_param in ss_abs_str_df['algo_param'].unique():
    sub_ss_df = ss_abs_str_df.query("algo_param == @algo_param")
    mr_count = sub_ss_df.query("mode == 'majority_rule'")['ext_mq_uniq'].sum()
    bc_count = sub_ss_df.query("mode == 'best_cluster'")['ext_mq_uniq'].sum()
    bm_count = sub_ss_df.query("mode == 'best_match'")['ext_mq_uniq'].sum()
    test_df = sub_ss_df.pivot(index='label_sample', columns='mode', values='ext_mq_uniq')

    # stats f_oneway functions takes the groups as input and returns ANOVA F and p value
    fvalue, pvalue = stats.f_oneway(test_df['majority_rule'],
                                    test_df['best_cluster'],
                                    test_df['best_match']
                                    )
    m_comp = pairwise_tukeyhsd(endog=sub_ss_df['ext_mq_uniq'], groups=sub_ss_df['mode'], alpha=0.05)
    stat, p = wilcoxon(test_df['majority_rule'], test_df['best_cluster'])

    print(f"\nThe Algorithm tested is {algo_param}")
    print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
    print(f"\nResults of Tukey HSD test:")
    print(m_comp)
    print(f"\nResults of Wilcoxon Signed-Rank Test:")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')
    stat, p = kruskal(test_df['majority_rule'],
                      test_df['best_cluster'],
                      test_df['best_match']
                      )
    print(f"\nResults of Kruskal-Wallis H Test:")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
    print(f"\nThe total number of NC bins for each mode is:\n")
    print(f"\t Majority Rule: {mr_count}")
    print(f"\t Best Cluster: {bc_count}")
    print(f"\t Best Match: {bm_count}")
    MR_count += mr_count
    BC_count += bc_count
    BM_count += bm_count

print(f"Total Majority Rule: {MR_count}")
print(f"Total Best Cluster: {BC_count}")
print(f"Total Best Match: {BM_count}")
lengs = len(ss_abs_str_df['algo_param'].unique())
print(f"Average Majority Rule: {MR_count / lengs}")
print(f"Average Best Cluster: {BC_count / lengs}")
print(f"Average Best Match: {BM_count / lengs}")

# Boxplots for mode and param set
ss_box = sns.catplot(x="mode", y="ext_mq_uniq", hue="algo",
                     col="param_set", col_wrap=2,
                     kind="box", data=ss_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.single.absolute.MQ.mode_param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Boxplots for mode
ss_box = sns.catplot(x="mode", y="ext_mq_uniq", hue="algo",
                     kind="box", data=ss_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.single.absolute.MQ.mode.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Boxplots for param set
ss_box = sns.catplot(x="param_set", y="ext_mq_uniq", hue="algo",
                     kind="box", data=ss_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.single.absolute.MQ.param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

########################################################################################################################
# Multi SABer
########################################################################################################################
print('############################################################')
print('# Multi SABer')
print('############################################################')

sm_df = pd.read_csv(saber_multi_file, header=0, sep='\t')
sm_df['label'] = [type2label[x] for x in sm_df['sample_type']]
sm_df['algo_rank'] = [algo2rank[x] for x in sm_df['algo']]
sm_df['type_rank'] = [type2rank[x] for x in sm_df['sample_type']]
sm_df['mode_rank'] = [mode2rank[x] for x in sm_df['mode']]
sm_df['param_rank'] = [param2rank[x] for x in sm_df['param_set']]
sm_df['mode_paramset'] = [str(x) + '_' + str(y) for x, y in
                          zip(sm_df['mode'], sm_df['param_set'])
                          ]
sm_df['algo_param'] = ['_'.join([str(x), str(y)]) for x, y in
                       zip(sm_df['algo'], sm_df['param_set'])
                       ]
sm_df['label_sample'] = ['_'.join([str(x), str(y)]) for x, y in
                         zip(sm_df['label'], sm_df['sample_id'])
                         ]
sm_abs_str_df = sm_df.query("level == 'strain_absolute'")
sm_abs_str_df.sort_values(by=['type_rank', 'algo_rank',
                              'mode_rank', 'param_set'
                              ], inplace=True)
ss_abs_str_median_df = sm_abs_str_df.groupby(['mode', 'algo']
                                             )[['ext_nc_uniq'
                                                ]].median().reset_index()
ss_abs_str_median_df.columns = ['mode', 'algo', 'median']
ss_abs_str_mean_df = sm_abs_str_df.groupby(['mode', 'algo']
                                           )[['ext_nc_uniq'
                                              ]].mean().reset_index()
ss_abs_str_mean_df.columns = ['mode', 'algo', 'mean']
ss_abs_str_std_df = sm_abs_str_df.groupby(['mode', 'algo']
                                          )[['ext_nc_uniq'
                                             ]].std().reset_index()
ss_abs_str_std_df.columns = ['mode', 'algo', 'stdev']
stats_df_list = [ss_abs_str_median_df, ss_abs_str_mean_df, ss_abs_str_std_df]
ss_abs_str_stats_df = reduce(lambda x, y: pd.merge(x, y, on=['mode', 'algo']), stats_df_list)
ss_abs_str_stats_df.sort_values(by='mean', ascending=False, inplace=True)

print(ss_abs_str_stats_df)

MR_count = 0
BC_count = 0
BM_count = 0
for algo_param in sm_abs_str_df['algo_param'].unique():
    sub_ss_df = sm_abs_str_df.query("algo_param == @algo_param")
    mr_count = sub_ss_df.query("mode == 'majority_rule'")['ext_nc_uniq'].sum()
    bc_count = sub_ss_df.query("mode == 'best_cluster'")['ext_nc_uniq'].sum()
    bm_count = sub_ss_df.query("mode == 'best_match'")['ext_nc_uniq'].sum()
    test_df = sub_ss_df.pivot(index='label_sample', columns='mode', values='ext_nc_uniq')

    # stats f_oneway functions takes the groups as input and returns ANOVA F and p value
    fvalue, pvalue = stats.f_oneway(test_df['majority_rule'],
                                    test_df['best_cluster'],
                                    test_df['best_match']
                                    )
    m_comp = pairwise_tukeyhsd(endog=sub_ss_df['ext_nc_uniq'], groups=sub_ss_df['mode'], alpha=0.05)
    stat, p = wilcoxon(test_df['majority_rule'], test_df['best_cluster'])

    print(f"\nThe Algorithm tested is {algo_param}")
    print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
    print(f"\nResults of Tukey HSD test:")
    print(m_comp)
    print(f"\nResults of Wilcoxon Signed-Rank Test:")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')
    stat, p = kruskal(test_df['majority_rule'],
                      test_df['best_cluster'],
                      test_df['best_match']
                      )
    print(f"\nResults of Kruskal-Wallis H Test:")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
    print(f"\nThe total number of NC bins for each mode is:\n")
    print(f"\t Majority Rule: {mr_count}")
    print(f"\t Best Cluster: {bc_count}")
    print(f"\t Best Match: {bm_count}")
    MR_count += mr_count
    BC_count += bc_count
    BM_count += bm_count

print(f"Total Majority Rule: {MR_count}")
print(f"Total Best Cluster: {BC_count}")
print(f"Total Best Match: {BM_count}")
lengs = len(sm_abs_str_df['algo_param'].unique())
print(f"Average Majority Rule: {MR_count / lengs}")
print(f"Average Best Cluster: {BC_count / lengs}")
print(f"Average Best Match: {BM_count / lengs}")

# Boxplots for mode and param set
ss_box = sns.catplot(x="mode", y="ext_nc_uniq", hue="algo",
                     col="param_set", col_wrap=2,
                     kind="box", data=sm_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.multi.absolute.NC.mode_param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Boxplots for mode
ss_box = sns.catplot(x="mode", y="ext_nc_uniq", hue="algo",
                     kind="box", data=sm_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.multi.absolute.NC.mode.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Boxplots for param set
ss_box = sns.catplot(x="param_set", y="ext_nc_uniq", hue="algo",
                     kind="box", data=sm_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.multi.absolute.NC.param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Medium Quality
ss_abs_str_median_df = sm_abs_str_df.groupby(['mode', 'algo']
                                             )[['ext_mq_uniq'
                                                ]].median().reset_index()
ss_abs_str_median_df.columns = ['mode', 'algo', 'median']
ss_abs_str_mean_df = sm_abs_str_df.groupby(['mode', 'algo']
                                           )[['ext_mq_uniq'
                                              ]].mean().reset_index()
ss_abs_str_mean_df.columns = ['mode', 'algo', 'mean']
ss_abs_str_std_df = sm_abs_str_df.groupby(['mode', 'algo']
                                          )[['ext_mq_uniq'
                                             ]].std().reset_index()
ss_abs_str_std_df.columns = ['mode', 'algo', 'stdev']
stats_df_list = [ss_abs_str_median_df, ss_abs_str_mean_df, ss_abs_str_std_df]
ss_abs_str_stats_df = reduce(lambda x, y: pd.merge(x, y, on=['mode', 'algo']), stats_df_list)
ss_abs_str_stats_df.sort_values(by='mean', ascending=False, inplace=True)

print(ss_abs_str_stats_df)

MR_count = 0
BC_count = 0
BM_count = 0
for algo_param in sm_abs_str_df['algo_param'].unique():
    sub_ss_df = sm_abs_str_df.query("algo_param == @algo_param")
    mr_count = sub_ss_df.query("mode == 'majority_rule'")['ext_mq_uniq'].sum()
    bc_count = sub_ss_df.query("mode == 'best_cluster'")['ext_mq_uniq'].sum()
    bm_count = sub_ss_df.query("mode == 'best_match'")['ext_mq_uniq'].sum()
    test_df = sub_ss_df.pivot(index='label_sample', columns='mode', values='ext_mq_uniq')

    # stats f_oneway functions takes the groups as input and returns ANOVA F and p value
    fvalue, pvalue = stats.f_oneway(test_df['majority_rule'],
                                    test_df['best_cluster'],
                                    test_df['best_match']
                                    )
    m_comp = pairwise_tukeyhsd(endog=sub_ss_df['ext_mq_uniq'], groups=sub_ss_df['mode'], alpha=0.05)
    stat, p = wilcoxon(test_df['majority_rule'], test_df['best_cluster'])

    print(f"\nThe Algorithm tested is {algo_param}")
    print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
    print(f"\nResults of Tukey HSD test:")
    print(m_comp)
    print(f"\nResults of Wilcoxon Signed-Rank Test:")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')
    stat, p = kruskal(test_df['majority_rule'],
                      test_df['best_cluster'],
                      test_df['best_match']
                      )
    print(f"\nResults of Kruskal-Wallis H Test:")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
    print(f"\nThe total number of NC bins for each mode is:\n")
    print(f"\t Majority Rule: {mr_count}")
    print(f"\t Best Cluster: {bc_count}")
    print(f"\t Best Match: {bm_count}")
    MR_count += mr_count
    BC_count += bc_count
    BM_count += bm_count

print(f"Total Majority Rule: {MR_count}")
print(f"Total Best Cluster: {BC_count}")
print(f"Total Best Match: {BM_count}")
lengs = len(sm_abs_str_df['algo_param'].unique())
print(f"Average Majority Rule: {MR_count / lengs}")
print(f"Average Best Cluster: {BC_count / lengs}")
print(f"Average Best Match: {BM_count / lengs}")

# Boxplots for mode and param set
ss_box = sns.catplot(x="mode", y="ext_mq_uniq", hue="algo",
                     col="param_set", col_wrap=2,
                     kind="box", data=sm_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.multi.absolute.MQ.mode_param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Boxplots for mode
ss_box = sns.catplot(x="mode", y="ext_mq_uniq", hue="algo",
                     kind="box", data=sm_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.multi.absolute.MQ.mode.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Boxplots for param set
ss_box = sns.catplot(x="param_set", y="ext_mq_uniq", hue="algo",
                     kind="box", data=sm_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.multi.absolute.MQ.param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()
'''

########################################################################################################################
# Single UniteM
########################################################################################################################
print('############################################################')
print('# Single UniteM')
print('############################################################')

us_df = pd.read_csv(unitem_single_file, header=0, sep='\t')
print(us_df.head())
print(us_df.columns)
us_df['label'] = [type2label[x] for x in us_df['sample_type']]
us_df['algo_rank'] = [algo2rank[x] for x in us_df['algo']]
us_df['type_rank'] = [type2rank[x] for x in us_df['sample_type']]
us_df['binner_rank'] = [binner2rank[x] for x in us_df['binner']]
us_df['label_sample'] = ['_'.join([str(x), str(y)]) for x, y in
                         zip(us_df['label'], us_df['sample_id'])
                         ]
us_abs_str_df = us_df.query("level == 'strain_absolute'")
us_abs_str_df.sort_values(by=['type_rank', 'algo_rank',
                              'binner_rank'
                              ], inplace=True)
us_abs_str_median_df = us_abs_str_df.groupby(['binner', 'algo']
                                             )[['ext_nc_uniq'
                                                ]].median().reset_index()
us_abs_str_median_df.columns = ['binner', 'algo', 'median']
us_abs_str_mean_df = us_abs_str_df.groupby(['binner', 'algo']
                                           )[['ext_nc_uniq'
                                              ]].mean().reset_index()
us_abs_str_mean_df.columns = ['binner', 'algo', 'mean']
us_abs_str_std_df = us_abs_str_df.groupby(['binner', 'algo']
                                          )[['ext_nc_uniq'
                                             ]].std().reset_index()
us_abs_str_std_df.columns = ['binner', 'algo', 'stdev']
stats_df_list = [us_abs_str_median_df, us_abs_str_mean_df,
                 us_abs_str_std_df
                 ]
us_abs_str_stats_df = reduce(lambda x, y:
                             pd.merge(x, y, on=['binner', 'algo']),
                             stats_df_list
                             )
us_abs_str_stats_df.sort_values(by='mean', ascending=False,
                                inplace=True
                                )

print(us_abs_str_stats_df)

test_df = us_abs_str_df.pivot(index='label_sample', columns='binner', values='ext_nc_uniq')
# stats f_oneway functions takes the groups as input and returns ANOVA F and p value
fvalue, pvalue = stats.f_oneway(test_df['maxbin_ms40'],
                                test_df['maxbin_ms107'],
                                test_df['metabat_specific'],
                                test_df['metabat_veryspecific'],
                                test_df['metabat_superspecific'],
                                test_df['metabat_sensitive'],
                                test_df['metabat_verysensitive'],
                                test_df['metabat2']
                                )
m_comp = pairwise_tukeyhsd(endog=us_abs_str_df['ext_nc_uniq'], groups=us_abs_str_df['binner'], alpha=0.05)
# stat, p = wilcoxon(test_df['majority_rule'], test_df['best_cluster'])

print(f"\nThe Algorithm tested is UniteM Binners")
print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
print(f"\nResults of Tukey HSD test:")
print(m_comp)
# print(f"\nResults of Wilcoxon Signed-Rank Test:")
# print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
# alpha = 0.05
# if p > alpha:
#    print('Same distribution (fail to reject H0)')
# else:
#    print('Different distribution (reject H0)')
stat, p = kruskal(test_df['maxbin_ms40'],
                  test_df['maxbin_ms107'],
                  test_df['metabat_specific'],
                  test_df['metabat_veryspecific'],
                  test_df['metabat_superspecific'],
                  test_df['metabat_sensitive'],
                  test_df['metabat_verysensitive'],
                  test_df['metabat2']
                  )
print(f"\nResults of Kruskal-Wallis H Test:")
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')
print(f"\nThe total number of NC bins for each binner is:\n")
print(us_abs_str_df.groupby(['binner'])['ext_nc_uniq'].sum())

# Boxplots for binner and param set
us_box = sns.catplot(x="label", y="ext_nc_uniq", hue="binner",
                     kind="box", data=us_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
us_box.savefig(os.path.join(workdir, 'UniteM.single.absolute.NC.binner.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Medium Quality
us_abs_str_median_df = us_abs_str_df.groupby(['binner', 'algo']
                                             )[['ext_mq_uniq'
                                                ]].median().reset_index()
us_abs_str_median_df.columns = ['binner', 'algo', 'median']
us_abs_str_mean_df = us_abs_str_df.groupby(['binner', 'algo']
                                           )[['ext_mq_uniq'
                                              ]].mean().reset_index()
us_abs_str_mean_df.columns = ['binner', 'algo', 'mean']
us_abs_str_std_df = us_abs_str_df.groupby(['binner', 'algo']
                                          )[['ext_mq_uniq'
                                             ]].std().reset_index()
us_abs_str_std_df.columns = ['binner', 'algo', 'stdev']
stats_df_list = [us_abs_str_median_df, us_abs_str_mean_df, us_abs_str_std_df]
us_abs_str_stats_df = reduce(lambda x, y: pd.merge(x, y, on=['binner', 'algo']), stats_df_list)
us_abs_str_stats_df.sort_values(by='mean', ascending=False, inplace=True)

print(us_abs_str_stats_df)

test_df = us_abs_str_df.pivot(index='label_sample', columns='binner', values='ext_mq_uniq')
# stats f_oneway functions takes the groups as input and returns ANOVA F and p value
fvalue, pvalue = stats.f_oneway(test_df['maxbin_ms40'],
                                test_df['maxbin_ms107'],
                                test_df['metabat_specific'],
                                test_df['metabat_veryspecific'],
                                test_df['metabat_superspecific'],
                                test_df['metabat_sensitive'],
                                test_df['metabat_verysensitive'],
                                test_df['metabat2']
                                )
m_comp = pairwise_tukeyhsd(endog=us_abs_str_df['ext_mq_uniq'], groups=us_abs_str_df['binner'], alpha=0.05)
# stat, p = wilcoxon(test_df['majority_rule'], test_df['best_cluster'])

print(f"\nThe Algorithm tested is UniteM Binners")
print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
print(f"\nResults of Tukey HSD test:")
print(m_comp)
# print(f"\nResults of Wilcoxon Signed-Rank Test:")
# print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
# alpha = 0.05
# if p > alpha:
#    print('Same distribution (fail to reject H0)')
# else:
#    print('Different distribution (reject H0)')
stat, p = kruskal(test_df['maxbin_ms40'],
                  test_df['maxbin_ms107'],
                  test_df['metabat_specific'],
                  test_df['metabat_veryspecific'],
                  test_df['metabat_superspecific'],
                  test_df['metabat_sensitive'],
                  test_df['metabat_verysensitive'],
                  test_df['metabat2']
                  )
print(f"\nResults of Kruskal-Wallis H Test:")
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')
print(f"\nThe total number of NC bins for each binner is:\n")
print(us_abs_str_df.groupby(['binner'])['ext_mq_uniq'].sum())

# Boxplots for binner and param set
us_box = sns.catplot(x="label", y="ext_mq_uniq", hue="binner",
                     kind="box", data=us_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
us_box.savefig(os.path.join(workdir, 'UniteM.single.absolute.MQ.binner.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()
