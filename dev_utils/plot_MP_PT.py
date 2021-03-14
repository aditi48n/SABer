import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_context("poster")
sns.set_style('whitegrid')
sns.set(font_scale=0.75)

sag_mp_file = 'E3_SAGs_MP.annotable.tsv'
sag_pt_file = 'E3_SAGs_PT.pwy.tsv'

sag_mp_df = pd.read_csv(sag_mp_file, sep='\t', header=0)
sag_pt_df = pd.read_csv(sag_pt_file, sep='\t', header=0)

sag_pt_df['sampleID'] = [x.strip('cyc') for x in sag_pt_df['pt_id']]
sag_mp_df['sampleID'] = [x.rsplit('_', 2)[0].lower() for x in sag_mp_df['orf_id']]

sag_mp_cnt_df = sag_mp_df.groupby(['sampleID']).count()
sag_pt_cnt_df = sag_pt_df.groupby(['sampleID']).count()

sag_merge_df = sag_mp_cnt_df.merge(sag_pt_cnt_df, on='sampleID', how='left')
keep_cols = ['orf_id', 'pwy_id', 'metacyc-2020-08-10(product)',
             'cazy-2020-06-01(product)', 'uniprot_sprot_2020-08-12(product)',
             'COG-14-2016-10-20(product)', 'refseq-nr-2020-08-12-rel-201(product)'
             ]
sag_trim_df = sag_merge_df[keep_cols]
sag_stack_df = sag_trim_df.stack().reset_index()
sag_stack_df.columns = ['sampleID', 'type', 'count']
col_dict = {'orf_id': 'ORFs', 'pwy_id': 'PathoPredict', 'metacyc-2020-08-10(product)': 'MetaCyc',
            'cazy-2020-06-01(product)': 'CAZy', 'uniprot_sprot_2020-08-12(product)': 'UniProt_Sprot',
            'COG-14-2016-10-20(product)': 'COG', 'refseq-nr-2020-08-12-rel-201(product)': 'RefSeq'
            }
sag_stack_df['type'] = [col_dict[x] for x in sag_stack_df['type']]
print(sag_stack_df.head())
g = sns.catplot(x='type', y='count', kind='box',  # hue='stage',
                data=sag_stack_df, linewidth=0.5
                )
g.set_xticklabels(rotation=30)

g.savefig("E3_MP_PT_box.png", bbox_inches='tight', dpi=300)
plt.clf()
plt.close()
