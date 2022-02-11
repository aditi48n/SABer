import pandas as pd

# Files and dirs
# denovo_out_file = glob.glob(joinpath(saberout_path, '*.denovo_clusters.tsv'))[0]
# trusted_out_file = glob.glob(joinpath(saberout_path, '*.hdbscan_clusters.tsv'))[0]
# ocsvm_out_file = glob.glob(joinpath(saberout_path, '*.ocsvm_clusters.tsv'))[0]
# inter_out_file = glob.glob(joinpath(saberout_path, '*.inter_clusters.tsv'))[0]


saber_single_file = '/home/ryan/SABer_local/benchmarking_output/errstat_inputs/SABer.single.errstat.tsv'
unitem_single_file = '/home/ryan/SABer_local/benchmarking_output/errstat_inputs/UniteM.single.errstat.tsv'
vamb_multi_file = '/home/ryan/SABer_local/benchmarking_output/errstat_inputs/VAMB.multi.errstat.tsv'

# Load stats tables
saber_single_df = pd.read_csv(saber_single_file, header=0, sep='\t')
unitem_single_df = pd.read_csv(unitem_single_file, header=0, sep='\t')
vamb_multi_df = pd.read_csv(vamb_multi_file, header=0, sep='\t')

bin_cat_df = pd.concat([saber_single_df, unitem_single_df,
                        vamb_multi_df
                        ])
mge_df = bin_cat_df.query("sample_type == 'MGE_6'")
print(mge_df.head())
