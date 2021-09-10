import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

score_file = '/home/ryan/Desktop/test_NMF/minhash_features/' \
             'CAMI_high_GoldStandardAssembly.nmf_scores.tsv'
score_df = pd.read_csv(score_file, sep='\t', header=0)
print(score_df.head())

denovo_file = '/home/ryan/Desktop/test_NMF/minhash_features/' \
              'denovo_id_map.tsv'
denovo_df = pd.read_csv(denovo_file, sep='\t', header=0)
print(denovo_df.head())

src2contig_file = '/home/ryan/Desktop/test_NMF/' \
                  'src2contig_map.tsv'
src2contig_df = pd.read_csv(src2contig_file, sep='\t', header=0)
print(src2contig_df.head())

score2denovo_df = score_df.merge(denovo_df, on='sag_id', how='left')
score2bp_df = score2denovo_df.merge(src2contig_df[['@@SEQUENCEID', 'bp_cnt']],
                                    left_on='contig_id', right_on='@@SEQUENCEID',
                                    how='left'
                                    )

score2bp_df['group1'] = score2bp_df['level'] + '_' + score2bp_df['inclusion']
score2bp_df['group2'] = score2bp_df['gamma'].astype(str) + '_' + score2bp_df['nu'].astype(str)
print(score2bp_df.shape)
# score_20k_df = score2bp_df.query('bp_cnt >= 20000')
# print(score_20k_df.shape)
# print(score_20k_df.head())
print(score2bp_df['bp_cnt'].min())
print(score2bp_df['bp_cnt'].max())

for grp1 in score2bp_df['group1'].unique():
    print(grp1)
    grp1_df = score2bp_df.query('group1 == @grp1 and precision != 0 and sensitivity != 0').copy()
    grp_bins = range(0, 4000000, 10000)
    grp_labels = range(10000, 4000000, 10000)
    grp1_df['bucket'] = pd.cut(grp1_df['bp_cnt'], bins=grp_bins, labels=grp_labels)
    '''
    max_list = []
    for sag_id in tqdm(grp1_df['sag_id'].unique()):
        sag_df = grp1_df.query('sag_id == @sag_id')
        sag_sort_df = sag_df.sort_values(['precision', 'sensitivity'], ascending=[False, False])
        p_max = sag_sort_df['precision'].values[0]
        r_max = sag_sort_df['sensitivity'].values[0]
        sag_max_df = sag_df.query('precision == @p_max and sensitivity == @r_max')
        #max_list.append(sag_max_df)
        max_list.append(sag_df)
        print(sag_df.head())
        sys.exit()
    max_df = pd.concat(max_list)
    '''
    df1 = pd.pivot_table(grp1_df, index=['group2', 'bucket'], values=['precision', 'sensitivity',
                                                                      'MCC'], aggfunc=np.mean
                         )
    df2 = pd.pivot_table(grp1_df, index=['group2', 'bucket'], values='sag_id',
                         aggfunc=len).rename(columns={'sag_id': 'count'}
                                             )
    df3 = pd.concat([df1, df2], axis=1).reset_index()
    df3['w_P'] = df3['precision'] * (df3['count'] / df3['count'].sum())
    df3['w_R'] = df3['sensitivity'] * (df3['count'] / df3['count'].sum())
    df3['w_MCC'] = df3['MCC'] * (df3['count'] / df3['count'].sum())
    df3['n_P'] = (df3['w_P'] - df3['w_P'].min()) / (df3['w_P'].max() - df3['w_P'].min())
    df3['n_R'] = (df3['w_R'] - df3['w_R'].min()) / (df3['w_R'].max() - df3['w_R'].min())
    df3['n_MCC'] = (df3['w_MCC'] - df3['w_MCC'].min()) / (df3['w_MCC'].max() - df3['w_MCC'].min())
    print(df3.head())
    ax = sns.relplot(x="bucket", y="precision", kind='line', data=df3)
    plt.savefig('/home/ryan/Desktop/test_NMF/minhash_features/bucket_' + grp1 + '_boxplot.png',
                bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()
    '''
    max_grp2_df = grp1_df.loc[grp1_df['group2'].isin(max_np_df.index.values)]
    print(max_grp2_df)
    
    #mean_grp2_df = pd.pivot_table(max_grp2_df, index=['group2'], values=['precision', 'sensitivity'],
    #                              aggfunc=np.mean
    #                              )
    mean_grp2_df = pd.pivot_table(df3, index=['group2'],
                                  values=['precision', 'sensitivity', 'n_P', 'n_R'],
                                  aggfunc=np.mean
                                  ).reset_index()
    print(mean_grp2_df)
    '''
