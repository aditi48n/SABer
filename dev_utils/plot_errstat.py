import os
import sys

import pandas as pd
import seaborn as sns

# specify that all columns should be shown
pd.set_option('max_columns', None)

# plot aestetics
sns.set_context("paper")

saber_single_file = sys.argv[1]
saber_multi_file = sys.argv[2]
unitem_single_file = sys.argv[3]
unitem_multi_file = sys.argv[4]
vamb_multi_file = sys.argv[5]

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
             'ocsvm': 2, 'intersect': 3,
             'xPG': 4
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
binnerconf2rank = {'maxbin_ms40': 0,
                   'maxbin_ms107': 1,
                   'metabat_specific': 2,
                   'metabat_veryspecific': 3,
                   'metabat_superspecific': 4,
                   'metabat_sensitive': 5,
                   'metabat_verysensitive': 6,
                   'metabat2': 7
                   }
binner2rank = {'maxbin': 0,
               'metabat': 1,
               'metabat2': 2,
               'VAMB': 3,
               'SABer_denovo': 4,
               'SABer_intersect': 5,
               'SABer_xPG': 6
               }

level2rank = {'exact_assembly_single': 0,
              'exact_absolute_single': 1,
              'strain_assembly_single': 2,
              'strain_absolute_single': 3,
              'exact_assembly_multi': 4,
              'exact_absolute_multi': 5,
              'strain_assembly_multi': 6,
              'strain_absolute_multi': 7
              }

# Load stats tables
saber_single_df = pd.read_csv(saber_single_file, header=0, sep='\t')
saber_multi_df = pd.read_csv(saber_multi_file, header=0, sep='\t')
unitem_single_df = pd.read_csv(unitem_single_file, header=0, sep='\t')
unitem_multi_df = pd.read_csv(unitem_multi_file, header=0, sep='\t')
vamb_multi_df = pd.read_csv(vamb_multi_file, header=0, sep='\t')

# Unify table formats
# col_order = ['binner', 'bin_mode', 'level', 'sample_type', 'sample_id',
#             'best_label', 'precision', 'sensitivity', 'MCC', 'F1',
#             'possible_bp', 'total_bp', 'exact_label', 'strain_label',
#             '>20Kb', 'NC_bins', 'MQ_bins'
#             ]
# SABer first
saber_single_df['binner'] = ['_'.join(['SABer', str(x), str(y), str(z)])
                             for x, y, z in
                             zip(saber_single_df['algorithm'],
                                 saber_single_df['mode'],
                                 saber_single_df['param_set']
                                 )
                             ]
saber_single_df['bin_mode'] = 'single'
saber_single_df['sample_id'] = ['S' + str(x) for x in
                                saber_single_df['sample_id']
                                ]
saber_s_df = saber_single_df  # .drop(columns=['algorithm', 'mode', 'param_set']
#      )[col_order]
saber_multi_df['binner'] = ['_'.join(['SABer', str(x), str(y), str(z)])
                            for x, y, z in
                            zip(saber_multi_df['algorithm'],
                                saber_multi_df['mode'],
                                saber_multi_df['param_set']
                                )
                            ]
saber_multi_df['bin_mode'] = 'multi'
saber_multi_df['sample_id'] = ['S' + str(x) for x in
                               saber_multi_df['sample_id']
                               ]
saber_m_df = saber_multi_df  # .drop(columns=['algorithm', 'mode', 'param_set']
#      )[col_order]

# UniteM Binners
unitem_single_df['bin_mode'] = 'single'
unitem_single_df['sample_id'] = ['S' + str(x) for x in
                                 unitem_single_df['sample_id']
                                 ]
unitem_s_df = unitem_single_df  # .drop(columns=['algorithm'])[col_order]
unitem_multi_df['bin_mode'] = 'multi'
unitem_multi_df['sample_id'] = ['S' + str(x) for x in
                                unitem_multi_df['sample_id']
                                ]
unitem_m_df = unitem_multi_df  # .drop(columns=['algorithm'])[col_order]

# VAMB
vamb_multi_df['bin_mode'] = 'multi'
vamb_multi_df['binner'] = 'VAMB'
vamb_multi_df['sample_id'] = ['S' + str(x) for x in
                              vamb_multi_df['sample_id']
                              ]
vamb_m_df = vamb_multi_df  #.drop(columns=['algorithm'])[col_order]

bin_cat_df = pd.concat([saber_s_df, saber_m_df,
                        unitem_s_df, unitem_m_df,
                        vamb_m_df
                        ])
bin_cat_df['binner_config'] = [x + '_' + y for x, y in zip(bin_cat_df['binner'],
                                                           bin_cat_df['bin_mode']
                                                           )]
bin_cat_df['level_mode'] = [x + '_' + y for x, y in zip(bin_cat_df['level'],
                                                        bin_cat_df['bin_mode']
                                                        )]
bin_cat_df['dataset'] = [type2label[x] for x in bin_cat_df['sample_type']]

########################################################################################################################
##### Calc all basic metrics ###########################################################################################
########################################################################################################################

# By dataset
cat_list = []
for binner in bin_cat_df['binner'].unique():
    for bin_mode in bin_cat_df['bin_mode'].unique():
        for dataset in bin_cat_df['dataset'].unique():
            for level in bin_cat_df['level'].unique():
                sub_err_df = bin_cat_df.query('binner == @binner & bin_mode == @bin_mode & '
                                              'dataset == @dataset & level == @level'
                                              )
                if sub_err_df.shape[0] != 0:
                    print(binner, bin_mode, dataset, level, sub_err_df.shape)
                    sub_err_df.sort_values(['precision', 'sensitivity'],
                                           ascending=[False, False], inplace=True
                                           )
                    sub_str_df = sub_err_df.drop_duplicates(subset='strain_label')
                    l_20 = '>20Kb'
                    ext_mq_df = sub_err_df.query("NC_bins == 'Yes' | MQ_bins == 'Yes' | @l_20 == 'Yes'")
                    ext_nc_df = sub_err_df.query("NC_bins == 'Yes' | @l_20 == 'Yes'")
                    str_mq_df = sub_str_df.query("NC_bins == 'Yes' | MQ_bins == 'Yes' | @l_20 == 'Yes'")
                    str_nc_df = sub_str_df.query("NC_bins == 'Yes' | @l_20 == 'Yes'")

                    mq_avg_mcc = ext_mq_df['MCC'].mean()
                    nc_avg_mcc = ext_nc_df['MCC'].mean()
                    mq_avg_p = ext_mq_df['precision'].mean()
                    nc_avg_p = ext_nc_df['precision'].mean()
                    mq_avg_r = ext_mq_df['sensitivity'].mean()
                    nc_avg_r = ext_nc_df['sensitivity'].mean()
                    ext_mq_cnt = ext_mq_df['MQ_bins'].count()
                    ext_nc_cnt = ext_nc_df['NC_bins'].count()
                    ext_mq_uniq = len(ext_mq_df['exact_label'].unique())
                    ext_nc_uniq = len(ext_nc_df['exact_label'].unique())
                    str_mq_cnt = str_mq_df['MQ_bins'].count()
                    str_nc_cnt = str_nc_df['NC_bins'].count()
                    str_mq_uniq = len(str_mq_df['strain_label'].unique())
                    str_nc_uniq = len(str_nc_df['strain_label'].unique())
                    sub_err_df['asm_per_bp'] = [x / y for x, y in
                                                zip(sub_err_df['possible_bp'],
                                                    sub_err_df['total_bp'])
                                                ]
                    sub_err_df['yes_NC'] = [1 if x >= 0.9 else 0 for x in sub_err_df['asm_per_bp']]
                    sub_err_df['yes_MQ'] = [1 if x >= 0.5 else 0 for x in sub_err_df['asm_per_bp']]
                    sub_err_df.sort_values(by='asm_per_bp', ascending=False, inplace=True)
                    poss_str_bp_df = sub_err_df[['strain_label', 'possible_bp',
                                                 'total_bp', 'asm_per_bp',
                                                 'yes_NC', 'yes_MQ'
                                                 ]].copy().drop_duplicates(subset='strain_label')
                    ext_mq_poss = sub_err_df['yes_MQ'].sum()
                    ext_nc_poss = sub_err_df['yes_NC'].sum()
                    str_mq_poss = poss_str_bp_df['yes_MQ'].sum()
                    str_nc_poss = poss_str_bp_df['yes_NC'].sum()
                    err_list = [binner, bin_mode, level, dataset, mq_avg_p,
                                mq_avg_r, mq_avg_mcc, nc_avg_p, nc_avg_r,
                                nc_avg_mcc, ext_mq_cnt, ext_mq_uniq,
                                ext_nc_cnt, ext_nc_uniq, str_mq_cnt, str_mq_uniq,
                                str_nc_cnt, str_nc_uniq, ext_mq_poss, ext_nc_poss,
                                str_mq_poss, str_nc_poss
                                ]
                    cat_list.append(err_list)
cat_cols = ['binner', 'bin_mode', 'level', 'dataset', 'mq_avg_p', 'mq_avg_r',
            'mq_avg_mcc', 'nc_avg_p', 'nc_avg_r', 'nc_avg_mcc', 'ext_mq_cnt',
            'ext_mq_uniq', 'ext_nc_cnt', 'ext_nc_uniq', 'str_mq_cnt',
            'str_mq_uniq', 'str_nc_cnt', 'str_nc_uniq', 'ext_mq_poss',
            'ext_nc_poss', 'str_mq_poss', 'str_nc_poss'
            ]
cat_metrics_df = pd.DataFrame(cat_list, columns=cat_cols)
cat_metrics_df.to_csv(os.path.join(workdir, 'ALL_BINNERS.dataset.avg_metrics.tsv'), sep='\t', index=False)

# By sample
cat_list = []
for binner in bin_cat_df['binner'].unique():
    for bin_mode in bin_cat_df['bin_mode'].unique():
        for dataset in bin_cat_df['dataset'].unique():
            for level in bin_cat_df['level'].unique():
                for sample_id in bin_cat_df['sample_id'].unique():
                    sub_err_df = bin_cat_df.query('binner == @binner & bin_mode == @bin_mode & '
                                                  'dataset == @dataset & level == @level & '
                                                  'sample_id == @sample_id'
                                                  )
                    if sub_err_df.shape[0] != 0:
                        print(binner, bin_mode, dataset, level, sample_id, sub_err_df.shape)
                        sub_err_df.sort_values(['precision', 'sensitivity'],
                                               ascending=[False, False], inplace=True
                                               )
                        sub_str_df = sub_err_df.drop_duplicates(subset='strain_label')
                        l_20 = '>20Kb'
                        ext_mq_df = sub_err_df.query("NC_bins == 'Yes' | MQ_bins == 'Yes' | @l_20 == 'Yes'")
                        ext_nc_df = sub_err_df.query("NC_bins == 'Yes' | @l_20 == 'Yes'")
                        str_mq_df = sub_str_df.query("NC_bins == 'Yes' | MQ_bins == 'Yes' | @l_20 == 'Yes'")
                        str_nc_df = sub_str_df.query("NC_bins == 'Yes' | @l_20 == 'Yes'")

                        mq_avg_mcc = ext_mq_df['MCC'].mean()
                        nc_avg_mcc = ext_nc_df['MCC'].mean()
                        mq_avg_p = ext_mq_df['precision'].mean()
                        nc_avg_p = ext_nc_df['precision'].mean()
                        mq_avg_r = ext_mq_df['sensitivity'].mean()
                        nc_avg_r = ext_nc_df['sensitivity'].mean()
                        ext_mq_cnt = ext_mq_df['MQ_bins'].count()
                        ext_nc_cnt = ext_nc_df['NC_bins'].count()
                        ext_mq_uniq = len(ext_mq_df['exact_label'].unique())
                        ext_nc_uniq = len(ext_nc_df['exact_label'].unique())
                        str_mq_cnt = str_mq_df['MQ_bins'].count()
                        str_nc_cnt = str_nc_df['NC_bins'].count()
                        str_mq_uniq = len(str_mq_df['strain_label'].unique())
                        str_nc_uniq = len(str_nc_df['strain_label'].unique())
                        sub_err_df['asm_per_bp'] = [x / y for x, y in
                                                    zip(sub_err_df['possible_bp'],
                                                        sub_err_df['total_bp'])
                                                    ]
                        sub_err_df['yes_NC'] = [1 if x >= 0.9 else 0 for x in sub_err_df['asm_per_bp']]
                        sub_err_df['yes_MQ'] = [1 if x >= 0.5 else 0 for x in sub_err_df['asm_per_bp']]
                        sub_err_df.sort_values(by='asm_per_bp', ascending=False, inplace=True)
                        poss_str_bp_df = sub_err_df[['strain_label', 'possible_bp',
                                                     'total_bp', 'asm_per_bp',
                                                     'yes_NC', 'yes_MQ'
                                                     ]].copy().drop_duplicates(subset='strain_label')
                        ext_mq_poss = sub_err_df['yes_MQ'].sum()
                        ext_nc_poss = sub_err_df['yes_NC'].sum()
                        str_mq_poss = poss_str_bp_df['yes_MQ'].sum()
                        str_nc_poss = poss_str_bp_df['yes_NC'].sum()
                        err_list = [binner, bin_mode, level, dataset, sample_id, mq_avg_p,
                                    mq_avg_r, mq_avg_mcc, nc_avg_p, nc_avg_r,
                                    nc_avg_mcc, ext_mq_cnt, ext_mq_uniq,
                                    ext_nc_cnt, ext_nc_uniq, str_mq_cnt, str_mq_uniq,
                                    str_nc_cnt, str_nc_uniq, ext_mq_poss, ext_nc_poss,
                                    str_mq_poss, str_nc_poss
                                    ]
                        cat_list.append(err_list)
cat_cols = ['binner', 'bin_mode', 'level', 'dataset', 'sample_id', 'mq_avg_p', 'mq_avg_r',
            'mq_avg_mcc', 'nc_avg_p', 'nc_avg_r', 'nc_avg_mcc', 'ext_mq_cnt',
            'ext_mq_uniq', 'ext_nc_cnt', 'ext_nc_uniq', 'str_mq_cnt',
            'str_mq_uniq', 'str_nc_cnt', 'str_nc_uniq', 'ext_mq_poss',
            'ext_nc_poss', 'str_mq_poss', 'str_nc_poss'
            ]
cat_metrics_df = pd.DataFrame(cat_list, columns=cat_cols)
cat_metrics_df.to_csv(os.path.join(workdir, 'ALL_BINNERS.sample.avg_metrics.tsv'), sep='\t', index=False)
sys.exit()
########################################################################################################################
##### RUN NC STATS #####################################################################################################
########################################################################################################################
print('############################################################')
print("RUN NC STATS")
print('############################################################')
cnt_df_list = []
for level_mode in bin_cat_df['level_mode'].unique():
    print('############################################################')
    print(f"The Level tested is {level_mode}")
    print('############################################################')
    sub_df = bin_cat_df.query("level_mode == @level_mode")
    # stats f_oneway functions takes the groups as input and returns ANOVA F and p value
    fvalue, pvalue = sci_stats.f_oneway(
        *(sub_df.loc[sub_df['binner_config'] == group, 'ext_nc_uniq']
          for group in sub_df['binner_config'].unique()
          ))
    m_comp = pairwise_tukeyhsd(endog=sub_df['ext_nc_uniq'],
                               groups=sub_df['binner_config'],
                               alpha=0.05
                               )
    print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
    print(f"\nResults of Tukey HSD test:")
    print(m_comp)
    stat, p = sci_stats.kruskal(
        *(sub_df.loc[sub_df['binner_config'] == group, 'ext_nc_uniq']
          for group in sub_df['binner_config'].unique()
          ))
    print(f"\nResults of Kruskal-Wallis H Test:")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
    count_nc_df = sub_df.groupby(['binner_config']
                                 )['ext_nc_uniq'].sum().reset_index()
    sorted_nc_df = count_nc_df.sort_values(by=['ext_nc_uniq'], ascending=False
                                           ).reset_index()
    sorted_nc_df['level_mode'] = level_mode
    cnt_df_list.append(sorted_nc_df)
    print(sorted_nc_df)

cat_cnt_df = pd.concat(cnt_df_list)
cat_cnt_df['binner'] = [x.split('_', 2)[0] + '_' + x.split('_', 2)[1]
                        if 'SABer' in x else x.split('_', 1)[0]
                        for x in cat_cnt_df['binner_config']
                        ]
filter_list = ['SABer_hdbscan', 'SABer_ocsvm']
filter_cnt_df = cat_cnt_df.query("binner not in @filter_list")
dedup_cnt_df = filter_cnt_df.drop_duplicates(subset=['binner', 'level_mode'])
print(dedup_cnt_df)
keep_binners_list = list(dedup_cnt_df['binner_config'])
keep_levmod_list = list(dedup_cnt_df['level_mode'])
print(keep_levmod_list)
print(keep_binners_list)
temp_cat_df = bin_cat_df.copy()
temp_cat_df['binner'] = [x.split('_', 2)[0] + '_' + x.split('_', 2)[1]
                         if 'SABer' in x else x.split('_', 1)[0]
                         for x in temp_cat_df['binner_config']
                         ]
temp_filter_df = temp_cat_df.query("binner not in @filter_list")
sub_binstat_df = temp_filter_df.query("binner_config in @keep_binners_list & "
                                      "level_mode in @keep_levmod_list"
                                      )
sub_binstat_df['bin_rank'] = [binner2rank[x] for x in
                              sub_binstat_df['binner']
                              ]
sub_binstat_df['type_rank'] = [type2rank[x] for x in
                               sub_binstat_df['sample_type']
                               ]
sub_binstat_df['level_rank'] = [level2rank[x] for x in
                                sub_binstat_df['level_mode']
                                ]
sub_binstat_df.sort_values(by=['level_rank', 'bin_rank', 'type_rank'
                               ], inplace=True
                           )
print(sub_binstat_df)
print(sub_binstat_df['binner'].unique())

# Boxplots for mode and param set
boxie = sns.catplot(x="dataset", y="ext_nc_uniq", hue="binner",
                    col="level_mode", col_wrap=2,
                    kind="box", data=sub_binstat_df, notch=True,
                    linewidth=0.75, saturation=0.75, width=0.75,
                    palette=sns.color_palette("muted")
                    )
boxie.savefig(os.path.join(workdir, 'ALL_BINNERS.NC.boxplot.png'),
              dpi=300
              )
plt.clf()
plt.close()

# Barplots for mode and param set
sum_binstat_df = sub_binstat_df.groupby(['binner', 'bin_rank',
                                         'type_rank', 'level_rank',
                                         'level_mode', 'dataset']
                                        )['ext_nc_uniq'].sum().reset_index()
sum_binstat_df.sort_values(by=['level_rank', 'bin_rank', 'type_rank'],
                           inplace=True)
barie = sns.catplot(x="dataset", y="ext_nc_uniq", hue="binner",
                    col="level_mode", col_wrap=2,
                    kind="bar", data=sum_binstat_df,
                    linewidth=0.75, saturation=0.75,
                    palette=sns.color_palette("muted")
                    )
barie.savefig(os.path.join(workdir, 'ALL_BINNERS.NC.barplot.png'),
              dpi=300
              )
plt.clf()
plt.close()

########################################################################################################################
##### RUN MQ STATS #####################################################################################################
########################################################################################################################
print('############################################################')
print("RUN MQ STATS")
print('############################################################')
cnt_df_list = []
for level_mode in bin_cat_df['level_mode'].unique():
    print('############################################################')
    print(f"The Level tested is {level_mode}")
    print('############################################################')
    sub_df = bin_cat_df.query("level_mode == @level_mode")
    # stats f_oneway functions takes the groups as input and returns ANOVA F and p value
    fvalue, pvalue = sci_stats.f_oneway(
        *(sub_df.loc[sub_df['binner_config'] == group, 'ext_mq_uniq']
          for group in sub_df['binner_config'].unique()
          ))
    m_comp = pairwise_tukeyhsd(endog=sub_df['ext_mq_uniq'],
                               groups=sub_df['binner_config'],
                               alpha=0.05
                               )
    print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
    print(f"\nResults of Tukey HSD test:")
    print(m_comp)
    stat, p = sci_stats.kruskal(
        *(sub_df.loc[sub_df['binner_config'] == group, 'ext_mq_uniq']
          for group in sub_df['binner_config'].unique()
          ))
    print(f"\nResults of Kruskal-Wallis H Test:")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
    count_nc_df = sub_df.groupby(['binner_config']
                                 )['ext_mq_uniq'].sum().reset_index()
    sorted_nc_df = count_nc_df.sort_values(by=['ext_mq_uniq'], ascending=False
                                           ).reset_index()
    sorted_nc_df['level_mode'] = level_mode
    cnt_df_list.append(sorted_nc_df)
    print(sorted_nc_df)

cat_cnt_df = pd.concat(cnt_df_list)
cat_cnt_df['binner'] = [x.split('_', 2)[0] + '_' + x.split('_', 2)[1]
                        if 'SABer' in x else x.split('_', 1)[0]
                        for x in cat_cnt_df['binner_config']
                        ]
filter_list = ['SABer_hdbscan', 'SABer_ocsvm']
filter_cnt_df = cat_cnt_df.query("binner not in @filter_list")
dedup_cnt_df = filter_cnt_df.drop_duplicates(subset=['binner', 'level_mode'])
print(dedup_cnt_df)
keep_binners_list = list(dedup_cnt_df['binner_config'])
keep_levmod_list = list(dedup_cnt_df['level_mode'])
print(keep_levmod_list)
print(keep_binners_list)
temp_cat_df = bin_cat_df.copy()
temp_cat_df['binner'] = [x.split('_', 2)[0] + '_' + x.split('_', 2)[1]
                         if 'SABer' in x else x.split('_', 1)[0]
                         for x in temp_cat_df['binner_config']
                         ]
temp_filter_df = temp_cat_df.query("binner not in @filter_list")
sub_binstat_df = temp_filter_df.query("binner_config in @keep_binners_list & "
                                      "level_mode in @keep_levmod_list"
                                      )
sub_binstat_df['bin_rank'] = [binner2rank[x] for x in
                              sub_binstat_df['binner']
                              ]
sub_binstat_df['type_rank'] = [type2rank[x] for x in
                               sub_binstat_df['sample_type']
                               ]
sub_binstat_df['level_rank'] = [level2rank[x] for x in
                                sub_binstat_df['level_mode']
                                ]
sub_binstat_df.sort_values(by=['level_rank', 'bin_rank', 'type_rank'
                               ], inplace=True
                           )
print(sub_binstat_df)
print(sub_binstat_df['binner'].unique())

# Boxplots for mode and param set
boxie = sns.catplot(x="dataset", y="ext_mq_uniq", hue="binner",
                    col="level_mode", col_wrap=2,
                    kind="box", data=sub_binstat_df, notch=True,
                    linewidth=0.75, saturation=0.75, width=0.75,
                    palette=sns.color_palette("muted")
                    )
boxie.savefig(os.path.join(workdir, 'ALL_BINNERS.MQ.boxplot.png'),
              dpi=300
              )
plt.clf()
plt.close()

# Barplots for mode and param set
sum_binstat_df = sub_binstat_df.groupby(['binner', 'bin_rank',
                                         'type_rank', 'level_rank',
                                         'level_mode', 'dataset']
                                        )['ext_mq_uniq'].sum().reset_index()
sum_binstat_df.sort_values(by=['level_rank', 'bin_rank', 'type_rank'],
                           inplace=True)
barie = sns.catplot(x="dataset", y="ext_mq_uniq", hue="binner",
                    col="level_mode", col_wrap=2,
                    kind="bar", data=sum_binstat_df,
                    linewidth=0.75, saturation=0.75,
                    palette=sns.color_palette("muted")
                    )
barie.savefig(os.path.join(workdir, 'ALL_BINNERS.MQ.barplot.png'),
              dpi=300
              )
plt.clf()
plt.close()

sys.exit()
########################################################################################################################
##### OLD CODE BELOW ###################################################################################################
########################################################################################################################


########################################################################################################################
# Single SABer - Near Complete, Absolute
########################################################################################################################
print('############################################################')
print('# Single SABer')
print('############################################################')

ss_df['label'] = [type2label[x] for x in ss_df['sample_type']]
ss_df['algo_rank'] = [algo2rank[x] for x in ss_df['algorithm']]
ss_df['type_rank'] = [type2rank[x] for x in ss_df['sample_type']]
ss_df['mode_rank'] = [mode2rank[x] for x in ss_df['mode']]
ss_df['param_rank'] = [param2rank[x] for x in ss_df['param_set']]
ss_df['mode_paramset'] = [str(x) + '_' + str(y) for x, y in
                          zip(ss_df['mode'], ss_df['param_set'])
                          ]
ss_df['algo_param'] = ['_'.join([str(x), str(y)]) for x, y in
                       zip(ss_df['algorithm'], ss_df['param_set'])
                       ]
ss_df['label_sample'] = ['_'.join([str(x), str(y)]) for x, y in
                         zip(ss_df['label'], ss_df['sample_id'])
                         ]
ss_abs_str_df = ss_df.query("level == 'strain_absolute'")
ss_abs_str_df.sort_values(by=['type_rank', 'algo_rank',
                              'mode_rank', 'param_set'
                              ], inplace=True)
ss_abs_str_median_df = ss_abs_str_df.groupby(['mode', 'algorithm']
                                             )[['ext_nc_uniq'
                                                ]].median().reset_index()
ss_abs_str_median_df.columns = ['mode', 'algorithm', 'median']
ss_abs_str_mean_df = ss_abs_str_df.groupby(['mode', 'algorithm']
                                           )[['ext_nc_uniq'
                                              ]].mean().reset_index()
ss_abs_str_mean_df.columns = ['mode', 'algorithm', 'mean']
ss_abs_str_std_df = ss_abs_str_df.groupby(['mode', 'algorithm']
                                          )[['ext_nc_uniq'
                                             ]].std().reset_index()
ss_abs_str_std_df.columns = ['mode', 'algorithm', 'stdev']
stats_df_list = [ss_abs_str_median_df, ss_abs_str_mean_df, ss_abs_str_std_df]
ss_abs_str_stats_df = reduce(lambda x, y: pd.merge(x, y, on=['mode', 'algorithm']), stats_df_list)
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
ss_box = sns.catplot(x="mode", y="ext_nc_uniq", hue="algorithm",
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
ss_box = sns.catplot(x="mode", y="ext_nc_uniq", hue="algorithm",
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
ss_box = sns.catplot(x="param_set", y="ext_nc_uniq", hue="algorithm",
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
ss_abs_str_median_df = ss_abs_str_df.groupby(['mode', 'algorithm']
                                             )[['ext_mq_uniq'
                                                ]].median().reset_index()
ss_abs_str_median_df.columns = ['mode', 'algorithm', 'median']
ss_abs_str_mean_df = ss_abs_str_df.groupby(['mode', 'algorithm']
                                           )[['ext_mq_uniq'
                                              ]].mean().reset_index()
ss_abs_str_mean_df.columns = ['mode', 'algorithm', 'mean']
ss_abs_str_std_df = ss_abs_str_df.groupby(['mode', 'algorithm']
                                          )[['ext_mq_uniq'
                                             ]].std().reset_index()
ss_abs_str_std_df.columns = ['mode', 'algorithm', 'stdev']
stats_df_list = [ss_abs_str_median_df, ss_abs_str_mean_df, ss_abs_str_std_df]
ss_abs_str_stats_df = reduce(lambda x, y: pd.merge(x, y, on=['mode', 'algorithm']), stats_df_list)
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
ss_box = sns.catplot(x="mode", y="ext_mq_uniq", hue="algorithm",
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
ss_box = sns.catplot(x="mode", y="ext_mq_uniq", hue="algorithm",
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
ss_box = sns.catplot(x="param_set", y="ext_mq_uniq", hue="algorithm",
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
sm_df['algo_rank'] = [algo2rank[x] for x in sm_df['algorithm']]
sm_df['type_rank'] = [type2rank[x] for x in sm_df['sample_type']]
sm_df['mode_rank'] = [mode2rank[x] for x in sm_df['mode']]
sm_df['param_rank'] = [param2rank[x] for x in sm_df['param_set']]
sm_df['mode_paramset'] = [str(x) + '_' + str(y) for x, y in
                          zip(sm_df['mode'], sm_df['param_set'])
                          ]
sm_df['algo_param'] = ['_'.join([str(x), str(y)]) for x, y in
                       zip(sm_df['algorithm'], sm_df['param_set'])
                       ]
sm_df['label_sample'] = ['_'.join([str(x), str(y)]) for x, y in
                         zip(sm_df['label'], sm_df['sample_id'])
                         ]
sm_abs_str_df = sm_df.query("level == 'strain_absolute'")
sm_abs_str_df.sort_values(by=['type_rank', 'algo_rank',
                              'mode_rank', 'param_set'
                              ], inplace=True)
ss_abs_str_median_df = sm_abs_str_df.groupby(['mode', 'algorithm']
                                             )[['ext_nc_uniq'
                                                ]].median().reset_index()
ss_abs_str_median_df.columns = ['mode', 'algorithm', 'median']
ss_abs_str_mean_df = sm_abs_str_df.groupby(['mode', 'algorithm']
                                           )[['ext_nc_uniq'
                                              ]].mean().reset_index()
ss_abs_str_mean_df.columns = ['mode', 'algorithm', 'mean']
ss_abs_str_std_df = sm_abs_str_df.groupby(['mode', 'algorithm']
                                          )[['ext_nc_uniq'
                                             ]].std().reset_index()
ss_abs_str_std_df.columns = ['mode', 'algorithm', 'stdev']
stats_df_list = [ss_abs_str_median_df, ss_abs_str_mean_df, ss_abs_str_std_df]
ss_abs_str_stats_df = reduce(lambda x, y: pd.merge(x, y, on=['mode', 'algorithm']), stats_df_list)
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
ss_box = sns.catplot(x="mode", y="ext_nc_uniq", hue="algorithm",
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
ss_box = sns.catplot(x="mode", y="ext_nc_uniq", hue="algorithm",
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
ss_box = sns.catplot(x="param_set", y="ext_nc_uniq", hue="algorithm",
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
ss_abs_str_median_df = sm_abs_str_df.groupby(['mode', 'algorithm']
                                             )[['ext_mq_uniq'
                                                ]].median().reset_index()
ss_abs_str_median_df.columns = ['mode', 'algorithm', 'median']
ss_abs_str_mean_df = sm_abs_str_df.groupby(['mode', 'algorithm']
                                           )[['ext_mq_uniq'
                                              ]].mean().reset_index()
ss_abs_str_mean_df.columns = ['mode', 'algorithm', 'mean']
ss_abs_str_std_df = sm_abs_str_df.groupby(['mode', 'algorithm']
                                          )[['ext_mq_uniq'
                                             ]].std().reset_index()
ss_abs_str_std_df.columns = ['mode', 'algorithm', 'stdev']
stats_df_list = [ss_abs_str_median_df, ss_abs_str_mean_df, ss_abs_str_std_df]
ss_abs_str_stats_df = reduce(lambda x, y: pd.merge(x, y, on=['mode', 'algorithm']), stats_df_list)
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
ss_box = sns.catplot(x="mode", y="ext_mq_uniq", hue="algorithm",
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
ss_box = sns.catplot(x="mode", y="ext_mq_uniq", hue="algorithm",
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
ss_box = sns.catplot(x="param_set", y="ext_mq_uniq", hue="algorithm",
                     kind="box", data=sm_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.multi.absolute.MQ.param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

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
us_df['algo_rank'] = [algo2rank[x] for x in us_df['algorithm']]
us_df['type_rank'] = [type2rank[x] for x in us_df['sample_type']]
us_df['binner_rank'] = [binner2rank[x] for x in us_df['binner']]
us_df['label_sample'] = ['_'.join([str(x), str(y)]) for x, y in
                         zip(us_df['label'], us_df['sample_id'])
                         ]
us_abs_str_df = us_df.query("level == 'strain_absolute'")
us_abs_str_df.sort_values(by=['type_rank', 'algo_rank',
                              'binner_rank'
                              ], inplace=True)
us_abs_str_median_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                             )[['ext_nc_uniq'
                                                ]].median().reset_index()
us_abs_str_median_df.columns = ['binner', 'algorithm', 'median']
us_abs_str_mean_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                           )[['ext_nc_uniq'
                                              ]].mean().reset_index()
us_abs_str_mean_df.columns = ['binner', 'algorithm', 'mean']
us_abs_str_std_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                          )[['ext_nc_uniq'
                                             ]].std().reset_index()
us_abs_str_std_df.columns = ['binner', 'algorithm', 'stdev']
stats_df_list = [us_abs_str_median_df, us_abs_str_mean_df,
                 us_abs_str_std_df
                 ]
us_abs_str_stats_df = reduce(lambda x, y:
                             pd.merge(x, y, on=['binner', 'algorithm']),
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
us_abs_str_median_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                             )[['ext_mq_uniq'
                                                ]].median().reset_index()
us_abs_str_median_df.columns = ['binner', 'algorithm', 'median']
us_abs_str_mean_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                           )[['ext_mq_uniq'
                                              ]].mean().reset_index()
us_abs_str_mean_df.columns = ['binner', 'algorithm', 'mean']
us_abs_str_std_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                          )[['ext_mq_uniq'
                                             ]].std().reset_index()
us_abs_str_std_df.columns = ['binner', 'algorithm', 'stdev']
stats_df_list = [us_abs_str_median_df, us_abs_str_mean_df, us_abs_str_std_df]
us_abs_str_stats_df = reduce(lambda x, y: pd.merge(x, y, on=['binner', 'algorithm']), stats_df_list)
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

########################################################################################################################
# Multi UniteM
########################################################################################################################
print('############################################################')
print('# Multi UniteM')
print('############################################################')

us_df = pd.read_csv(unitem_multi_file, header=0, sep='\t')
print(us_df.head())
print(us_df.columns)
us_df['label'] = [type2label[x] for x in us_df['sample_type']]
us_df['algo_rank'] = [algo2rank[x] for x in us_df['algorithm']]
us_df['type_rank'] = [type2rank[x] for x in us_df['sample_type']]
us_df['binner_rank'] = [binner2rank[x] for x in us_df['binner']]
us_df['label_sample'] = ['_'.join([str(x), str(y)]) for x, y in
                         zip(us_df['label'], us_df['sample_id'])
                         ]
us_abs_str_df = us_df.query("level == 'strain_absolute'")
us_abs_str_df.sort_values(by=['type_rank', 'algo_rank',
                              'binner_rank'
                              ], inplace=True)
us_abs_str_median_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                             )[['ext_nc_uniq'
                                                ]].median().reset_index()
us_abs_str_median_df.columns = ['binner', 'algorithm', 'median']
us_abs_str_mean_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                           )[['ext_nc_uniq'
                                              ]].mean().reset_index()
us_abs_str_mean_df.columns = ['binner', 'algorithm', 'mean']
us_abs_str_std_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                          )[['ext_nc_uniq'
                                             ]].std().reset_index()
us_abs_str_std_df.columns = ['binner', 'algorithm', 'stdev']
stats_df_list = [us_abs_str_median_df, us_abs_str_mean_df,
                 us_abs_str_std_df
                 ]
us_abs_str_stats_df = reduce(lambda x, y:
                             pd.merge(x, y, on=['binner', 'algorithm']),
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
us_box.savefig(os.path.join(workdir, 'UniteM.multi.absolute.NC.binner.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Medium Quality
us_abs_str_median_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                             )[['ext_mq_uniq'
                                                ]].median().reset_index()
us_abs_str_median_df.columns = ['binner', 'algorithm', 'median']
us_abs_str_mean_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                           )[['ext_mq_uniq'
                                              ]].mean().reset_index()
us_abs_str_mean_df.columns = ['binner', 'algorithm', 'mean']
us_abs_str_std_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                          )[['ext_mq_uniq'
                                             ]].std().reset_index()
us_abs_str_std_df.columns = ['binner', 'algorithm', 'stdev']
stats_df_list = [us_abs_str_median_df, us_abs_str_mean_df, us_abs_str_std_df]
us_abs_str_stats_df = reduce(lambda x, y: pd.merge(x, y, on=['binner', 'algorithm']), stats_df_list)
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
us_box.savefig(os.path.join(workdir, 'UniteM.multi.absolute.MQ.binner.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

