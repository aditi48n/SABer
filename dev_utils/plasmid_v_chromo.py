import glob
from os.path import join as joinpath

import pandas as pd

# specify that all columns should be shown
pd.set_option('max_columns', None)

# Input file

# Files and dirs
saberout_path = '/home/ryan/SABer_local/benchmarking_output/' \
                'single_binner_bench/MGE_6/SABer_single/0/' \
                '*/*/'
denovo_file_list = glob.glob(joinpath(saberout_path, '*.denovo_clusters.tsv'))
trusted_file_list = glob.glob(joinpath(saberout_path, '*.hdbscan_clusters.tsv'))
ocsvm_file_list = glob.glob(joinpath(saberout_path, '*.ocsvm_clusters.tsv'))
inter_file_list = glob.glob(joinpath(saberout_path, '*.inter_clusters.tsv'))

saber_single_file = '/home/ryan/SABer_local/benchmarking_output/errstat_inputs/SABer.single.errstat.tsv'
unitem_single_file = '/home/ryan/SABer_local/benchmarking_output/errstat_inputs/UniteM.single.errstat.tsv'
vamb_multi_file = '/home/ryan/SABer_local/benchmarking_output/errstat_inputs/VAMB.multi.errstat.tsv'

# Load stats tables
saber_single_df = pd.read_csv(saber_single_file, header=0, sep='\t')
saber_single_df['binner'] = ['_'.join(['SABer', str(x), str(y), str(z)])
                             for x, y, z in
                             zip(saber_single_df['algorithm'],
                                 saber_single_df['mode'],
                                 saber_single_df['param_set']
                                 )
                             ]
saber_mge_df = saber_single_df.query("sample_type == 'MGE_6'")
saber_mge_df.rename(columns={'>20Kb': 'over20Kb'}, inplace=True)
saber_mg_df = saber_mge_df.query("over20Kb == 'Yes' & "
                                 "MQ_bins == 'Yes'"
                                 )
print(saber_mg_df.head())
flurp
unitem_single_df = pd.read_csv(unitem_single_file, header=0, sep='\t')
unitem_mge_df = unitem_single_df.query("sample_type == 'MGE_6'")

vamb_multi_df = pd.read_csv(vamb_multi_file, header=0, sep='\t')
vamb_multi_df['binner'] = 'VAMB'
vamb_mge_df = vamb_multi_df.query("sample_type == 'MGE_6'")

bin_cat_df = pd.concat([saber_single_df, unitem_single_df,
                        vamb_multi_df
                        ])
mge_df = bin_cat_df.query("sample_type == 'MGE_6'")
print(mge_df.head())
