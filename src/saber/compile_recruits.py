import logging
from os.path import join as o_join

import pandas as pd

import utilities as s_utils


def run_combine_recruits(xpg_path, mg_file, clusters):
    denovo_clust_df = clusters[0]
    trusted_clust_df = clusters[1]

    logging.info('Combining All Recruits\n')
    mg_contigs_dict = s_utils.get_seqs(mg_file)
    mg_contigs = tuple([(r.name, r.seq) for r in mg_contigs_dict])
    mg_contigs_df = pd.DataFrame(mg_contigs, columns=['contig_id', 'seq'])
    # De Novo Bins
    for best_label in set(denovo_clust_df['best_label']):
        sub_merge_df = denovo_clust_df[['best_label', 'contig_id']
        ].query('best_label == @best_label').drop_duplicates()
        logging.info('Recruited %s contigs from entire analysis for %s\n' %
                     (sub_merge_df.shape[0], best_label)
                     )
        final_rec = o_join(xpg_path, str(best_label) + '.xPG.fasta')
        with open(final_rec, 'w') as final_out:
            denovo_contig_list = list(set(sub_merge_df['contig_id']))
            mg_sub_filter_df = mg_contigs_df.query('contig_id in @denovo_contig_list')
            final_mgsubs_list = ['\n'.join(['>' + x[0], x[1]]) for x in
                                 zip(mg_sub_filter_df['contig_id'],
                                     mg_sub_filter_df['seq']
                                     )
                                 ]
            final_out.write('\n'.join(final_mgsubs_list))
    # Trusted Bins
    if isinstance(trusted_clust_df, pd.DataFrame):
        for best_label in set(trusted_clust_df['best_label']):
            sub_merge_df = trusted_clust_df.query('best_label == @best_label')
            logging.info('Recruited %s contigs from entire analysis for %s\n' %
                         (sub_merge_df.shape[0], best_label)
                         )
            final_rec = o_join(xpg_path, str(best_label) + '.xPG.fasta')
            with open(final_rec, 'w') as final_out:
                trust_contig_list = list(set(sub_merge_df['contig_id']))
                mg_sub_filter_df = mg_contigs_df.query('contig_id in @trust_contig_list')
                final_mgsubs_list = ['\n'.join(['>' + x[0], x[1]]) for x in
                                     zip(mg_sub_filter_df['contig_id'],
                                         mg_sub_filter_df['seq']
                                         )
                                     ]
                final_out.write('\n'.join(final_mgsubs_list))
