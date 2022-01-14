import logging
from os.path import join as o_join
from subprocess import Popen

import pandas as pd

import utilities as s_utils


def run_combine_recruits(save_dirs_dict, mg_file, clusters, trusted_list, threads):
    denovo_clust_df = clusters[0]
    trusted_clust_df = clusters[1]
    ocsvm_clust_df = clusters[2]
    inter_clust_df = clusters[3]
    denovo_sv_path = save_dirs_dict['denovo']
    hdbscan_sv_path = save_dirs_dict['hdbscan']
    ocsvm_sv_path = save_dirs_dict['ocsvm']
    inter_sv_path = save_dirs_dict['intersect']
    xpg_sv_path = save_dirs_dict['xpgs']

    logging.info('Combining All Recruits\n')
    mg_contigs_dict = s_utils.get_seqs(mg_file)
    mg_contigs = tuple([(r.name, r.seq) for r in mg_contigs_dict])
    mg_contigs_df = pd.DataFrame(mg_contigs, columns=['contig_id', 'seq'])
    # De Novo Bins
    for best_label in set(denovo_clust_df['best_label']):
        sub_merge_df = denovo_clust_df[['best_label', 'contig_id']
        ].query('best_label == @best_label').drop_duplicates()
        logging.info('Recruited %s contigs from De Novo analysis for %s\n' %
                     (sub_merge_df.shape[0], best_label)
                     )
        final_rec = o_join(denovo_sv_path, str(best_label) + '.denovo.fasta')
        with open(final_rec, 'w') as final_out:
            contig_list = list(set(sub_merge_df['contig_id']))
            mg_sub_filter_df = mg_contigs_df.query('contig_id in @contig_list')
            final_mgsubs_list = ['\n'.join(['>' + x[0], x[1]]) for x in
                                 zip(mg_sub_filter_df['contig_id'],
                                     mg_sub_filter_df['seq']
                                     )
                                 ]
            final_out.write('\n'.join(final_mgsubs_list))

    # HDBSCAN Bins
    for best_label in set(trusted_clust_df['best_label']):
        sub_merge_df = trusted_clust_df[['best_label', 'contig_id']
        ].query('best_label == @best_label').drop_duplicates()
        logging.info('Recruited %s contigs from HDBSCAN anchored analysis for %s\n' %
                     (sub_merge_df.shape[0], best_label)
                     )
        final_rec = o_join(hdbscan_sv_path, str(best_label) + '.hdbscan.fasta')
        with open(final_rec, 'w') as final_out:
            contig_list = list(set(sub_merge_df['contig_id']))
            mg_sub_filter_df = mg_contigs_df.query('contig_id in @contig_list')
            final_mgsubs_list = ['\n'.join(['>' + x[0], x[1]]) for x in
                                 zip(mg_sub_filter_df['contig_id'],
                                     mg_sub_filter_df['seq']
                                     )
                                 ]
            final_out.write('\n'.join(final_mgsubs_list))
    # OC-SVM Bins
    for best_label in set(ocsvm_clust_df['best_label']):
        sub_merge_df = ocsvm_clust_df[['best_label', 'contig_id']
        ].query('best_label == @best_label').drop_duplicates()
        logging.info('Recruited %s contigs from OC-SVM anchored analysis for %s\n' %
                     (sub_merge_df.shape[0], best_label)
                     )
        final_rec = o_join(ocsvm_sv_path, str(best_label) + '.ocsvm.fasta')
        with open(final_rec, 'w') as final_out:
            contig_list = list(set(sub_merge_df['contig_id']))
            mg_sub_filter_df = mg_contigs_df.query('contig_id in @contig_list')
            final_mgsubs_list = ['\n'.join(['>' + x[0], x[1]]) for x in
                                 zip(mg_sub_filter_df['contig_id'],
                                     mg_sub_filter_df['seq']
                                     )
                                 ]
            final_out.write('\n'.join(final_mgsubs_list))
    # Combined Bins
    if isinstance(inter_clust_df, pd.DataFrame):
        for best_label in set(inter_clust_df['best_label']):
            sub_merge_df = inter_clust_df.query('best_label == @best_label')
            logging.info('Recruited %s contigs from intersection of anchored analysis for %s\n' %
                         (sub_merge_df.shape[0], best_label)
                         )
            final_rec = o_join(inter_sv_path, str(best_label) + '.bin.fasta')
            with open(final_rec, 'w') as final_out:
                contig_list = list(set(sub_merge_df['contig_id']))
                mg_sub_filter_df = mg_contigs_df.query('contig_id in @contig_list')
                final_mgsubs_list = ['\n'.join(['>' + x[0], x[1]]) for x in
                                     zip(mg_sub_filter_df['contig_id'],
                                         mg_sub_filter_df['seq']
                                         )
                                     ]
                final_out.write('\n'.join(final_mgsubs_list))
        # Combine final recruits and reference trusted contigs
        trust_dict = {t[0]: t[1] for t in trusted_list}
        for t_id in set(inter_clust_df['best_label']):
            t_file = trust_dict[t_id]
            concat_file = o_join(xpg_sv_path, t_id + '.concat.fasta')
            with open(concat_file, 'w') as cat_out:
                data = []
                with open(t_file, 'r') as t_in:
                    data.extend(t_in.readlines())
                recruit_bin = o_join(inter_sv_path, t_id + '.bin.fasta')
                with open(recruit_bin, 'r') as r_file:
                    data.extend(r_file.readlines())
                join_data = '\n'.join(data).replace('\n\n', '\n')
                cat_out.write(join_data)

            # Use BBTools dedupe.sh to deduplicate the extend SAG file
            dedupe_fa = o_join(xpg_sv_path, t_id + '.xPG.fasta')
            dedupe_cmd = ['dedupe.sh', 'in=' + concat_file, 'out=' + dedupe_fa,
                          'threads=' + str(threads), 'minidentity=97', 'overwrite=true']
            logging.info('Running BBtools dedup on %s\n' % t_id)
            run_mem = Popen(dedupe_cmd)
            run_mem.communicate()
