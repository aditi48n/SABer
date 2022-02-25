import glob
import os

import pandas as pd

# specify that all columns should be shown
pd.set_option('max_columns', None)

cwd_dir = "/home/ryan/Desktop/HQ_stuff"

barrnap_file_list = glob.glob(
    os.path.join(cwd_dir, 'HQ_barrnap/scratch/st-shallam-1/mcglock/'
                          'SI60/SABer_output/SI060_*/*/*/barrnap/*.gff'
                 ))

bar_df_list = []
for bar_file in barrnap_file_list:
    print(bar_file)
    try:
        sample_id = bar_file.split('/')[11]
        mode = bar_file.split('/')[12]
        set = bar_file.split('/')[13]
        sag_id = bar_file.split('/')[15].split('.', 1)[0]
        filter_len = bar_file.split('/')[15].split('.')[4]
        bar_df = pd.read_csv(bar_file, sep='\t', header=None,
                             skiprows=1
                             )
        bar_df.columns = ['contig_id', 'software_version',
                          'seq_type', 'start', 'end',
                          'e-value', 'unknown1',
                          'unknown2', 'description'
                          ]
        bar_df['sample_id'] = sample_id
        bar_df['mode'] = mode
        bar_df['set'] = set
        bar_df['sag_id'] = sag_id
        bar_df['filter_len'] = filter_len
        bar_df['subunit'] = [x.split(';', 1)[0].split('=', 1)[1].split('_', 1)[0]
                             for x in bar_df['description']
                             ]
        bar_df['product'] = [x.split(';', 1)[1].split('=', 1)[1]
                             for x in bar_df['description']
                             ]
        bar_df['notes'] = [x.split(';note=', 1)[1] if 'note' in x
                           else '' for x in bar_df['description']
                           ]
        bar_df['completeness'] = ['full' if x is '' else 'partial'
                                  for x in bar_df['notes']
                                  ]
        bar_df['percent'] = [100 if x is '' else
                             int(x.split('only ', 1
                                         )[1].split(' percent', 1)[0]
                                 )
                             for x in bar_df['notes']
                             ]
        bar_df_list.append(bar_df)
    except:
        print('Somthing wrong with gff file, probably empty...')

bar_concat_df = pd.concat(bar_df_list)
bar_concat_df.to_csv(os.path.join(cwd_dir, 'barrnap_all_output.tsv'),
                     sep='\t', index=False
                     )
'''
bar_concat_df = pd.read_csv(os.path.join(cwd_dir, 'barrnap_output.tsv'),
                            sep='\t', header=0
                            )
'''
bar_trim_df = bar_concat_df[['contig_id', 'sample_id', 'mode', 'set',
                             'sag_id', 'filter_len', 'subunit',
                             'completeness', 'percent'
                             ]]
index_list = ['sample_id', 'mode', 'set', 'sag_id', 'filter_len']
cols = 'subunit'
vals = 'percent'
bar_piv_df = pd.pivot_table(values=vals, index=index_list,
                            columns=cols, data=bar_trim_df
                            ).reset_index().fillna(0)
bar_piv_df.rename(columns={"16S": "rRNA_16S",
                           "23S": "rRNA_23S",
                           "5S": "rRNA_5S"
                           }, inplace=True)

hq_full_df = bar_piv_df.query("rRNA_16S == 100 &"
                              "rRNA_23S == 100 &"
                              "rRNA_5S == 100"
                              )
print(hq_full_df.head())
print(hq_full_df.shape)
print(len(hq_full_df['sag_id'].unique()))
hq_full_df.to_csv(os.path.join(cwd_dir, 'barrnap_HQ_output.tsv'),
                  sep='\t', index=False
                  )
