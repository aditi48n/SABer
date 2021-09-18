import logging
from os.path import join as o_join

import pandas as pd

import utilities as s_utils


def run_combine_recruits(xpg_path, mg_file, tetra_df_dict,
                         minhash_dict, sag_list
                         ):
    logging.info('Combining All Recruits\n')
    mg_contigs_dict = s_utils.get_seqs(mg_file)
    mg_contigs = tuple([(r.name, r.seq) for r in mg_contigs_dict])
    tetra_df = tetra_df_dict['comb']
    # Merge MinHash and GMM Tetra (passed first by ABR)
    minhash_df = minhash_dict[51]
    minhash_filter_df = minhash_df.loc[minhash_df['jacc_sim'] >= 0.90]
    mh_gmm_merge_df = minhash_filter_df[['sag_id', 'contig_id']
    ].merge(tetra_df[['sag_id', 'contig_id']],
            how='outer', on=['sag_id', 'contig_id']
            ).drop_duplicates()
    mh_gmm_merge_df.to_csv(o_join(xpg_path, 'CONTIG_MAP.xPG.tsv'), sep='\t', index=False)
    mg_contigs_df = pd.DataFrame(mg_contigs, columns=['contig_id', 'seq'])
    '''
    for sag_id in set(mh_gmm_merge_df['sag_id']):
        sub_merge_df = mh_gmm_merge_df.loc[mh_gmm_merge_df['sag_id'] == sag_id]
        logging.info('Recruited %s contigs from entire analysis for %s\n' %
                     (sub_merge_df.shape[0], sag_id)
                     )
        final_rec = o_join(xpg_path, sag_id + '.xPG.fasta')
        with open(final_rec, 'w') as final_out:
            mg_sub_filter_df = mg_contigs_df.loc[mg_contigs_df['contig_id'
            ].isin(sub_merge_df['contig_id'])
            ]
            final_mgsubs_list = ['\n'.join(['>' + x[0], x[1]]) for x in
                                 zip(mg_sub_filter_df['contig_id'],
                                     mg_sub_filter_df['seq']
                                     )
                                 ]
            final_out.write('\n'.join(final_mgsubs_list))
    '''
    # merge de novo config into proper xPGs
    contig_set = list(set(mh_gmm_merge_df['contig_id']))
    single_list = []
    while len(contig_set) != 0:
        print(len(contig_set))
        contig_id = contig_set[0]
        contig_df = mh_gmm_merge_df.loc[mh_gmm_merge_df['contig_id'] == contig_id]
        recruit_df = mh_gmm_merge_df.loc[mh_gmm_merge_df['sag_id'].isin(contig_df['sag_id'])]
        if recruit_df.shape[0] > 1:
            logging.info('Recruited %s contigs from entire analysis for %s\n' %
                         (len(set(recruit_df['contig_id'])), contig_id)
                         )
            final_rec = o_join(xpg_path, contig_id.replace('|', '_') + '.xPG.fasta')
            with open(final_rec, 'w') as final_out:
                mg_sub_filter_df = mg_contigs_df.loc[mg_contigs_df['contig_id'
                ].isin(recruit_df['contig_id'])
                ]
                final_mgsubs_list = ['\n'.join(['>' + x[0], x[1]]) for x in
                                     zip(mg_sub_filter_df['contig_id'],
                                         mg_sub_filter_df['seq']
                                         )
                                     ]
                final_out.write('\n'.join(final_mgsubs_list))
            contig_del_list = [contig_id] + list(set(recruit_df['contig_id']))
            contig_set = [x for x in contig_set if x not in contig_del_list]
        else:
            contig_del_list = [contig_id]
            contig_set = [x for x in contig_set if x not in contig_del_list]
            single_list.append(contig_id)
