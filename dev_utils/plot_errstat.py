import os
import sys
from functools import reduce

import pandas as pd
import seaborn as sns

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

# Munge SABer single/mulit
# Near Complete Single
# Absolute as reference
ss_df = pd.read_csv(saber_single_file, header=0, sep='\t')
ss_df['label'] = [type2label[x] for x in ss_df['sample_type']]
ss_df['algo_rank'] = [algo2rank[x] for x in ss_df['algo']]
ss_df['type_rank'] = [type2rank[x] for x in ss_df['sample_type']]
ss_df['mode_rank'] = [mode2rank[x] for x in ss_df['mode']]
ss_df['param_rank'] = [param2rank[x] for x in ss_df['param_set']]
ss_df['mode_paramset'] = [x + '_' + y for x, y in
                          zip(ss_df['mode'], ss_df['param_set'])
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

ss_mode_max_df = ss_abs_str_stats_df.groupby(['mode', 'algo']
                                             )[['mean'
                                                ]].max().reset_index()
print(ss_abs_str_stats_df)
print(ss_mode_max_df)
sys.exit()

ss_mode_max_df.sort_values(by='ext_nc_uniq', ascending=False, inplace=True)
ss_best_mode_df = ss_mode_max_df.drop_duplicates(subset='algo')
print(ss_best_mode_df)
print(ss_best_mode_df.shape)
sys.exit()
# Boxplots for mode and param set
ss_box = sns.catplot(x="mode", y="ext_nc_uniq", hue="algo",
                     col="param_set", col_wrap=2,
                     kind="box", data=ss_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.single.absolute.mode_param.boxplot.png'),
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
ss_box.savefig(os.path.join(workdir, 'SABer.single.absolute.mode.boxplot.png'),
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
ss_box.savefig(os.path.join(workdir, 'SABer.single.absolute.param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

sys.exit()

# Assembly as reference
ss_ass_str_df = ss_df.query("level == 'strain_assembly'")
ss_ass_str_df.sort_values(by=['type_rank', 'algo_rank',
                              'mode_rank', 'param_set'
                              ], inplace=True)
# Boxplots for mode and param set
ss_box = sns.catplot(x="mode", y="ext_nc_uniq", hue="algo",
                     col="param_set", col_wrap=2,
                     kind="box", data=ss_ass_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.single.assembly.mode_param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Near Complete Multi
# Absolute as reference
sm_df = pd.read_csv(saber_multi_file, header=0, sep='\t')
sm_df['label'] = [type2label[x] for x in sm_df['sample_type']]
sm_df['algo_rank'] = [algo2rank[x] for x in sm_df['algo']]
sm_df['type_rank'] = [type2rank[x] for x in sm_df['sample_type']]
sm_df['mode_rank'] = [mode2rank[x] for x in sm_df['mode']]
sm_df['param_rank'] = [param2rank[x] for x in sm_df['param_set']]
sm_df['mode_paramset'] = [x + '_' + y for x, y in
                          zip(sm_df['mode'], sm_df['param_set'])
                          ]
sm_abs_str_df = sm_df.query("level == 'strain_absolute'")
sm_abs_str_df.sort_values(by=['type_rank', 'algo_rank'], inplace=True)
sm_abs_str_df.sort_values(by=['type_rank', 'algo_rank',
                              'mode_rank', 'param_set'
                              ], inplace=True)
sm_box = sns.catplot(x="mode", y="ext_nc_uniq", hue="algo",
                     col="param_set", col_wrap=2,
                     kind="box", data=sm_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
sm_box.savefig(os.path.join(workdir, 'SABer.multi.absolute.mode_param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Assembly as reference
sm_ass_str_df = sm_df.query("level == 'strain_assembly'")
sm_ass_str_df.sort_values(by=['type_rank', 'algo_rank',
                              'mode_rank', 'param_set'
                              ], inplace=True)
# Boxplots for mode and param set
sm_box = sns.catplot(x="mode", y="ext_nc_uniq", hue="algo",
                     col="param_set", col_wrap=2,
                     kind="box", data=sm_ass_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
sm_box.savefig(os.path.join(workdir, 'SABer.multi.assembly.mode_param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

sys.exit()

# Build boxplots
ss_box = sns.catplot(x="label", y="ext_nc_uniq", hue="algo",
                     kind="box", data=ss_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.single.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()
sm_box = sns.catplot(x="label", y="ext_nc_uniq", hue="algo",
                     kind="box", data=sm_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
sm_box.savefig(os.path.join(workdir, 'SABer.multi.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

print(ss_df.head())
print(sm_df.head())
