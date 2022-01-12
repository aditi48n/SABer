import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_context("poster")
sns.set_style('whitegrid')
sns.set(font_scale=0.75)
paired_cols = sns.color_palette("Paired")
set2_cols = sns.color_palette("Set2")

# SAGs
sag_mp_file = 'E3_SAGs_MP.annotable.tsv'
sag_pt_file = 'PGDB_Pathways_SAGs_noprune.tsv'
sag_mp_df = pd.read_csv(sag_mp_file, sep='\t', header=0)
sag_pt_df = pd.read_csv(sag_pt_file, sep='\t', header=0)
sag_pt_df = sag_pt_df.set_index('sampleID').stack().reset_index()
sag_pt_df.columns = ['sampleID', 'pwy_id', 'predicted']
sag_pt_df = sag_pt_df.loc[sag_pt_df['predicted'] == 1]
sag_pt_df['sampleID'] = [x.strip('cyc') for x in sag_pt_df['sampleID']]
# sag_pt_df['pwy_id'] = [x.split(' ', 1)[0].replace('\"', '').replace('(', '').replace(')', '') for x in sag_pt_df['pwy_id']]
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
sag_stack_df.columns = ['sampleID', 'annotype', 'count']
col_dict = {'orf_id': 'ORFs', 'pwy_id': 'PathoLogic', 'metacyc-2020-08-10(product)': 'MetaCyc',
            'cazy-2020-06-01(product)': 'CAZy', 'uniprot_sprot_2020-08-12(product)': 'UniProt_Sprot',
            'COG-14-2016-10-20(product)': 'COG', 'refseq-nr-2020-08-12-rel-201(product)': 'RefSeq'
            }
sag_stack_df['annotype'] = [col_dict[x] for x in sag_stack_df['annotype']]
sag_stack_df['datatype'] = 'SAG'
sag_stack_df['strain'] = [x.rsplit('_', 1)[0] for x in sag_stack_df['sampleID']]
col_order = ['ORFs', 'RefSeq', 'COG', 'MetaCyc', 'UniProt_Sprot', 'CAZy', 'PathoLogic']
sag_trim_df.to_csv('E3_MP_PT_SAGs.tsv', sep='\t')

# xPGs
xpg_mp_file = 'E3_xPGs_MP.annotable.tsv'
xpg_pt_file = 'PGDB_Pathways_xPGs_noprune.tsv'
xpg_mp_df = pd.read_csv(xpg_mp_file, sep='\t', header=0)
xpg_pt_df = pd.read_csv(xpg_pt_file, sep='\t', header=0)
xpg_pt_df = xpg_pt_df.set_index('sampleID').stack().reset_index()
xpg_pt_df.columns = ['sampleID', 'pwy_id', 'predicted']
xpg_pt_df = xpg_pt_df.loc[xpg_pt_df['predicted'] == 1]
xpg_pt_df['sampleID'] = [x.rsplit('_', 1)[0] for x in xpg_pt_df['sampleID']]
# xpg_pt_df['pwy_id'] = [x.split(' ', 1)[0].replace('\"', '').replace('(', '').replace(')', '') for x in xpg_pt_df['pwy_id']]
xpg_mp_df['sampleID'] = [x.rsplit('_', 3)[0].lower() for x in xpg_mp_df['orf_id']]
xpg_mp_cnt_df = xpg_mp_df.groupby(['sampleID']).count()
xpg_pt_cnt_df = xpg_pt_df.groupby(['sampleID']).count()
xpg_merge_df = xpg_mp_cnt_df.merge(xpg_pt_cnt_df, on='sampleID', how='left')
xpg_trim_df = xpg_merge_df[keep_cols]
xpg_stack_df = xpg_trim_df.stack().reset_index()
xpg_stack_df.columns = ['sampleID', 'annotype', 'count']
xpg_stack_df['annotype'] = [col_dict[x] for x in xpg_stack_df['annotype']]
xpg_stack_df['datatype'] = 'xPG'
xpg_stack_df['strain'] = [x.rsplit('_', 1)[0] for x in xpg_stack_df['sampleID']]
col_order = ['ORFs', 'RefSeq', 'COG', 'MetaCyc', 'UniProt_Sprot', 'CAZy', 'PathoLogic']
xpg_trim_df.to_csv('E3_MP_PT_xPGs.tsv', sep='\t')

# SRCs
src_mp_file = 'E3_SRCs_MP.annotable.tsv'
src_pt_file = 'PGDB_Pathways_SRCs_noprune.tsv'
src_mp_df = pd.read_csv(src_mp_file, sep='\t', header=0)
src_pt_df = pd.read_csv(src_pt_file, sep='\t', header=0)
src_pt_df = src_pt_df.set_index('sampleID').stack().reset_index()
src_pt_df.columns = ['sampleID', 'pwy_id', 'predicted']
src_pt_df = src_pt_df.loc[src_pt_df['predicted'] == 1]
src_pt_df['sampleID'] = [x.strip('cyc') for x in src_pt_df['sampleID']]
# src_pt_df['pwy_id'] = [x.split(' ', 1)[0].replace('\"', '').replace('(', '').replace(')', '') for x in src_pt_df['pwy_id']]
src_mp_df['sampleID'] = [x.rsplit('_', 2)[0].lower() for x in src_mp_df['orf_id']]
src_mp_cnt_df = src_mp_df.groupby(['sampleID']).count()
src_pt_cnt_df = src_pt_df.groupby(['sampleID']).count()
src_merge_df = src_mp_cnt_df.merge(src_pt_cnt_df, on='sampleID', how='left')
src_trim_df = src_merge_df[keep_cols]
src_stack_df = src_trim_df.stack().reset_index()
src_stack_df.columns = ['sampleID', 'annotype', 'count']
src_stack_df['annotype'] = [col_dict[x] for x in src_stack_df['annotype']]
src_stack_df['datatype'] = 'Genome'
src_stack_df['strain'] = [x for x in src_stack_df['sampleID']]
src_trim_df.to_csv('E3_MP_PT_srcs.tsv', sep='\t')

# METACYC
mcyc_pt_file = 'E3_SRCs_MetaCyc.pwy.tsv'
mcyc_pt_df = pd.read_csv(mcyc_pt_file, sep='\t', header=0)
mcyc_pt_df['sampleID'] = [x.strip('cyc') for x in mcyc_pt_df['sampleID']]
# mcyc_pt_df['pwy_id'] = [x.split(' ', 1)[0].replace('\"', '').replace('(', '').replace(')', '') for x in mcyc_pt_df['pwy_id']]
mcyc_pt_cnt_df = mcyc_pt_df.groupby(['sampleID']).count()
mcyc_pt_cnt_df['orf_id'] = 0
mcyc_pt_cnt_df['metacyc-2020-08-10(product)'] = 0
mcyc_pt_cnt_df['cazy-2020-06-01(product)'] = 0
mcyc_pt_cnt_df['uniprot_sprot_2020-08-12(product)'] = 0
mcyc_pt_cnt_df['COG-14-2016-10-20(product)'] = 0
mcyc_pt_cnt_df['refseq-nr-2020-08-12-rel-201(product)'] = 0
mcyc_trim_df = mcyc_pt_cnt_df[keep_cols]
mcyc_stack_df = mcyc_trim_df.stack().reset_index()
mcyc_stack_df.columns = ['sampleID', 'annotype', 'count']
mcyc_stack_df['annotype'] = [col_dict[x] for x in mcyc_stack_df['annotype']]
mcyc_stack_df['datatype'] = 'MetaCyc'
mcyc_stack_df['strain'] = [x for x in mcyc_stack_df['sampleID']]

# CONCAT AND PLOT STUFF
col_order = ['ORFs', 'RefSeq', 'COG', 'MetaCyc', 'UniProt_Sprot', 'CAZy', 'PathoLogic']
concat_stack_df = pd.concat([mcyc_stack_df, src_stack_df, sag_stack_df, xpg_stack_df])
# print(concat_stack_df.head())
no_metacyc_df = concat_stack_df.loc[concat_stack_df['datatype'] != 'MetaCyc']
no_metacyc_df.to_csv('E3_MP_PT_stack.tsv', sep='\t', index=False)
g = sns.catplot(x='annotype', y='count', kind='bar', hue='datatype',
                data=no_metacyc_df, linewidth=0.5, order=col_order,
                ci=95, col='strain',
                palette=['darkgray', paired_cols[0], paired_cols[7]]
                )
g.set_xticklabels(rotation=30)
g.savefig("E3_MP_PT_cat_noprune.png", bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

patho_df = concat_stack_df.loc[concat_stack_df['annotype'] == 'PathoLogic']
# print(patho_df.head())
g = sns.catplot(x='strain', y='count', kind='bar', hue='datatype',
                data=patho_df, linewidth=0.5, ci=95,
                palette=['lightgray', 'darkgray', paired_cols[0], paired_cols[7]]
                )
# g.set(ylim=(350, 500))
g.set_xticklabels(rotation=30)
g.savefig("E3_MP_PT_strain_box_noprune.png", bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

patho_df = concat_stack_df.loc[concat_stack_df['annotype'] == 'PathoLogic']
# print(patho_df.head())
g = sns.catplot(x='strain', y='count', kind='bar', hue='datatype',
                data=patho_df, linewidth=0.5, ci=95,
                palette=['lightgray', 'darkgray', paired_cols[0], paired_cols[7]]
                )
for r in g.ax.patches:
    x, y = r.get_xy()
    h = int(r.get_height())
    g.ax.text(x, h + 8, h)
g.set_xticklabels(rotation=30)
g.savefig("E3_MP_PT_strain_box_labels_noprune.png", bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

# STATS
mcyc_pt_df['genomeID'] = [x for x in mcyc_pt_df['sampleID']]
mcyc_pt_df['datatype'] = 'MetaCyc'

src_pt_df['genomeID'] = [x for x in src_pt_df['sampleID']]
src_pt_df['datatype'] = 'genome'

sag_pt_df['genomeID'] = [x.rsplit('_', 1)[0] for x in sag_pt_df['sampleID']]
sag_pt_df['datatype'] = 'SAG'

xpg_pt_df['genomeID'] = [x.rsplit('_', 1)[0] for x in xpg_pt_df['sampleID']]
xpg_pt_df['datatype'] = 'xPG'

# df_list = [src_pt_df, sag_pt_df, xpg_pt_df]
df_list = [sag_pt_df, xpg_pt_df]
stats_list = []
for df in df_list:
    for sampleID in df['sampleID'].unique():
        pt_df = df.loc[df['sampleID'] == sampleID]
        genomeID = pt_df['genomeID'].unique()[0]
        datatype = pt_df['datatype'].unique()[0]
        g_mcyc_pt_df = src_pt_df.loc[src_pt_df['genomeID'] == genomeID]
        metacyc_pwy = len(set(g_mcyc_pt_df['pwy_id']))
        src_pwy = len(set(pt_df['pwy_id']))
        meta_src_pwy = len(set(g_mcyc_pt_df['pwy_id']).intersection(set(pt_df['pwy_id'])))
        meta_src_pwy_non = src_pwy - meta_src_pwy
        P_src = (meta_src_pwy / src_pwy) * 100
        R_src = (meta_src_pwy / metacyc_pwy) * 100
        stats_list.append([sampleID, datatype, genomeID, P_src, R_src])

    stats_df = pd.DataFrame(stats_list, columns=['sampleID', 'datatype', 'genomeID', 'Precision', 'Completeness'])

stats_df = stats_df.loc[((stats_df['Precision'] != 0) & (stats_df['Completeness'] != 0))]
count_dict = {'SAG': stats_df.loc[stats_df['datatype'] == 'SAG'].shape[0],
              'xPG': stats_df.loc[stats_df['datatype'] == 'xPG'].shape[0]
              }
stats_df['datatype'] = [x + ' (n = ' + str(count_dict[x]) + ')' for x in stats_df['datatype']]
g = sns.scatterplot(x='Completeness', y='Precision', hue='datatype', data=stats_df, s=100)
# g.set_xticklabels(rotation=30)
fig = g.get_figure()
fig.savefig("E3_MP_PT_stats_noprune_scatter.png", bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

stats_df.set_index(['sampleID', 'datatype', 'genomeID'], inplace=True)
stack_stats_df = stats_df.stack().reset_index()
stack_stats_df.columns = ['sampleID', 'datatype', 'genomeID', 'metric', 'score']
print(stack_stats_df)
stats_df.to_csv('E3_MP_PT_stats_noprune.tsv', sep='\t')

g = sns.catplot(x='genomeID', y='score', kind='bar', hue='datatype',
                col='metric',
                data=stack_stats_df, linewidth=0.5, ci=95,
                palette=[paired_cols[0], paired_cols[7]]
                )
g.set_xticklabels(rotation=30)
g.savefig("E3_MP_PT_stats_noprune_box_SRC_R.png", bbox_inches='tight', dpi=300)
plt.clf()
plt.close()
# print(src_pt_df.head())
# print(sag_pt_df.head())
# print(xpg_pt_df.head())
