import sys

import pandas as pd
import plotly.graph_objects as go

pd.set_option('display.max_columns', None)

cv_file = sys.argv[1]
cv_df = pd.read_csv(cv_file, sep='\t', header=0)
sv_dir = '/home/ryan/Desktop/SABer_CV/'
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
