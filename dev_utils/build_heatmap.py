import glob
import os
import sys

import pandas as pd

# import plotly.graph_objects as go

pd.set_option('display.max_columns', None)

working_dir = '/home/ryan/Desktop/SABer_CV/'
sample_list = glob.glob(os.path.join(working_dir, "*/CV_*.tsv"))
############################################################################################
cv_concat_list = []
cv_all_list = []
config_list = [['hdbscan', 'denovo'], ['hdbscan', 'hdbscan'], ['ocsvm', 'ocsvm']]
for sample in sample_list:
    cv_df = pd.read_csv(sample, sep='\t', header=0)
    if len(cv_df.columns) == 16:
        col_list = list(cv_df.columns)
        col_list[-1] = 'sample_id'
        cv_df.columns = col_list
        sample_id = sample.rsplit('/')[-2].split('_', 1)[1]
        cv_df['sample_id'] = [sample_id + '_' + str(x) for x in cv_df['sample_id']]
    else:
        sample_id = sample.rsplit('/')[-2].split('_', 1)[1]
        cv_df['sample_id'] = sample_id
    # get best performing configs
    for config in config_list:
        cv_algo = config[0]
        algo = config[1]
        config_df = cv_df.query("cv_algo == @cv_algo & algo == @algo")
        for level in config_df['level'].unique():
            level_df = config_df.query("level == @level")
            nc_max = level_df['nc_cnt'].max()
            mq_max = level_df['mq_cnt'].max()
            nc_df = level_df.query("nc_cnt == @nc_max")
            mq_df = level_df.query("mq_cnt == @mq_max")
            nc_mq_df = nc_df.query("mq_cnt == @mq_max")
            mq_nc_df = mq_df.query("nc_cnt == @nc_max")
            if nc_mq_df.shape[0] != 0:
                cv_concat_list.append(nc_mq_df)
            else:
                cv_concat_list.append(nc_df)

            if mq_nc_df.shape[0] != 0:
                cv_concat_list.append(mq_nc_df)
            else:
                cv_concat_list.append(mq_df)
    cv_all_list.append(cv_df)

cv_concat_df = pd.concat(cv_concat_list)
cv_concat_df.to_csv(os.path.join(working_dir, "CV_best_table.tsv"), sep='\t', index=False)

cv_all_df = pd.concat(cv_all_list)
cv_all_df.to_csv(os.path.join(working_dir, "CV_all_table.tsv"), sep='\t', index=False)

cluster_file = '/home/ryan/Desktop/renyi_entropy/references/cluster_table.tsv'
cluster_df = pd.read_csv(cluster_file, sep='\t', header=0)
clust_cv_df = cv_concat_df.merge(cluster_df[['sample_id', 'sample_type', 'cluster', 'best_match']],
                                 on='sample_id', how='left'
                                 )
clust_cv_df.drop(['mq_avg_p', 'mq_avg_r', 'mq_avg_mcc',
                  'nc_avg_p', 'nc_avg_r', 'nc_avg_mcc'
                  ], axis=1, inplace=True
                 )
clust_cv_df.drop_duplicates(inplace=True)

clust_all_df = cv_all_df.merge(cluster_df[['sample_id', 'sample_type', 'cluster', 'best_match']],
                               on='sample_id', how='left'
                               )
clust_all_df.drop(['mq_avg_p', 'mq_avg_r', 'mq_avg_mcc',
                   'nc_avg_p', 'nc_avg_r', 'nc_avg_mcc'
                   ], axis=1, inplace=True
                  )
clust_all_df.drop_duplicates(inplace=True)
agg_list = []
for config in config_list:
    cv_algo = config[0]
    algo = config[1]
    config_df = clust_cv_df.query("cv_algo == @cv_algo & algo == @algo")
    group_list = ['sample_type', 'cluster', 'sample_id', 'best_match']
    for group in group_list:
        nc_max_list = []
        mq_max_list = []
        for type in config_df[group].unique():
            type_df = config_df.loc[config_df[group] == type]
            nc_max = type_df['nc_cnt'].max()
            mq_max = type_df['mq_cnt'].max()
            nc_max_df = type_df.query('nc_cnt == @nc_max')
            mq_max_df = type_df.query('mq_cnt == @mq_max')
            type_nc_cnt_df = nc_max_df.groupby(['cv_algo', 'cv_param1', 'cv_param2', 'cv_val1',
                                                'cv_val2', 'algo', 'level', group]
                                               )['nc_cnt'].count().reset_index()
            type_mq_cnt_df = mq_max_df.groupby(['cv_algo', 'cv_param1', 'cv_param2', 'cv_val1',
                                                'cv_val2', 'algo', 'level', group]
                                               )['mq_cnt'].count().reset_index()
            for i, row in type_nc_cnt_df.iterrows():
                g_type = row[group]
                cv_algo = row['cv_algo']
                cv_param1 = row['cv_param1']
                cv_param2 = row['cv_param2']
                cv_val1 = row['cv_val1']
                cv_val2 = row['cv_val2']
                algo = row['algo']
                level = row['level']
                sub_type_df = clust_all_df.loc[clust_all_df[group] == g_type]
                sub_cv_df = sub_type_df.query("cv_algo == @cv_algo &"
                                              "cv_param1 == @cv_param1 &"
                                              "cv_param2 == @cv_param2 &"
                                              "cv_val1 == @cv_val1 &"
                                              "cv_val2 == @cv_val2 &"
                                              "algo == @algo &"
                                              "level == @level"
                                              )
                nc_sum_df = sub_cv_df.groupby([group, 'cv_algo', 'cv_param1', 'cv_param2',
                                               'cv_val1', 'cv_val2', 'algo', 'level']
                                              )['nc_cnt'].sum().reset_index()
                mq_sum_df = sub_cv_df.groupby([group, 'cv_algo', 'cv_param1', 'cv_param2',
                                               'cv_val1', 'cv_val2', 'algo', 'level']
                                              )['mq_cnt'].sum().reset_index()

                nc_max_list.append(nc_sum_df)
                mq_max_list.append(mq_sum_df)
        nc_best_df = pd.concat(nc_max_list)
        nc_best_df.drop_duplicates(subset=[group], inplace=True)
        mq_best_df = pd.concat(mq_max_list)
        mq_best_df.drop_duplicates(subset=[group], inplace=True)
        nc_agg = nc_best_df['nc_cnt'].sum()
        mq_agg = mq_best_df['mq_cnt'].sum()
        agg_list.append([group, cv_algo, algo, nc_agg, mq_agg])

agg_df = pd.DataFrame(agg_list, columns=['grouping', 'cv_algo', 'algo', 'nc_agg', 'mq_agg'])
print(agg_df)

# Assign the best match params to the SI data
real_dir = '/home/ryan/Desktop/renyi_entropy/SI/'
best_df = pd.read_csv(os.path.join(real_dir, 'cluster_table.tsv'), sep='\t', header=0)
best_df = best_df[['sample_type', 'sample_id', 'best_match']]
best_df.columns = ['sample', 'depth', 'best_match']
best_cv_df = best_df.merge(cv_concat_df[['cv_algo', 'cv_param1', 'cv_param2', 'cv_val1',
                                         'cv_val2', 'algo', 'level', 'sample_id']],
                           left_on='best_match', right_on='sample_id', how='left'
                           )
print(best_cv_df.head())
hdbscan_df = best_cv_df.query("cv_algo == 'hdbscan'")
ocsvm_df = best_cv_df.query("cv_algo == 'ocsvm'")
sort_hdb_df = hdbscan_df.sort_values(['cv_val1', 'cv_val2'], ascending=[True, True])
sort_oc_df = ocsvm_df.sort_values(['cv_val1', 'cv_val2'], ascending=[False, True])
dedup_hdb_df = sort_hdb_df.drop_duplicates(subset=['sample', 'depth', 'best_match',
                                                   'best_match', 'cv_algo', 'cv_param1',
                                                   'cv_param2', 'algo', 'level']
                                           )
dedup_oc_df = sort_oc_df.drop_duplicates(subset=['sample', 'depth', 'best_match',
                                                 'best_match', 'cv_algo', 'cv_param1',
                                                 'cv_param2', 'algo', 'level']
                                         )
filter_df = pd.concat([dedup_hdb_df, dedup_oc_df])
filter_df.to_csv(os.path.join(real_dir, 'params_table.tsv'), sep='\t', index=False)

sys.exit()
cv_file = sys.argv[1]
cv_df = pd.read_csv(cv_file, sep='\t', header=0)
sv_dir = os.path.split(cv_file)[0] + '/'
for cv_algo in cv_df['cv_algo'].unique():
    cv_algo_df = cv_df.query('cv_algo == @cv_algo')
    for algo in cv_algo_df['algo'].unique():
        algo_df = cv_algo_df.query('algo == @algo')
        if ((cv_algo == 'hdbscan' and (algo == 'hdbscan' or algo == 'denovo')) or
                (cv_algo == 'ocsvm' and algo == 'ocsvm')
        ):
            for level in algo_df['level'].unique():
                level_df = algo_df.query('level == @level')
                fig = go.Figure(data=go.Heatmap(
                    z=level_df['mq_cnt'],
                    x=[str(x) for x in level_df['cv_val1']],
                    y=[str(x) for x in level_df['cv_val2']],
                    hoverongaps=False, colorscale='Viridis'))
                sv_file = sv_dir + '_'.join([cv_algo, algo, level]) + '.MQ_BINS.png'
                fig.write_image(sv_file)
                fig2 = go.Figure(data=go.Heatmap(
                    z=level_df['nc_cnt'],
                    x=[str(x) for x in level_df['cv_val1']],
                    y=[str(x) for x in level_df['cv_val2']],
                    hoverongaps=False, colorscale='Viridis'))
                sv2_file = sv_dir + '_'.join([cv_algo, algo, level]) + '.NC_BINS.png'
                fig2.write_image(sv2_file)
                '''
                fig3 = go.Figure(data=go.Heatmap(
                    z=level_df['mq_avg_mcc'],
                    x=[str(x) for x in level_df['cv_val1']],
                    y=[str(x) for x in level_df['cv_val2']],
                    hoverongaps=False, colorscale='Viridis'))
                sv3_file = sv_dir + '_'.join([cv_algo, algo, level]) + '.MQ_MCC_BINS.png'
                fig3.write_image(sv3_file)
                fig4 = go.Figure(data=go.Heatmap(
                    z=level_df['nc_avg_mcc'],
                    x=[str(x) for x in level_df['cv_val1']],
                    y=[str(x) for x in level_df['cv_val2']],
                    hoverongaps=False, colorscale='Viridis'))
                sv4_file = sv_dir + '_'.join([cv_algo, algo, level]) + '.NC_MCC_BINS.png'
                fig4.write_image(sv4_file)
                '''
