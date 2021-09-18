import argparse
import logging
import multiprocessing
from os.path import isfile, getsize
from os.path import join as o_join

import numpy as np
import pandas as pd
import sourmash
from sourmash.lca import LCA_Database
from sourmash.sbtmh import SigLeaf

import logger as s_log
import utilities as s_utils

pd.set_option('display.max_columns', None)


def run_minhash_recruiter(sig_path, mhr_path, sag_sub_files, mg_sub_file, nthreads, force):
    logging.info('Starting MinHash Recruitment\n')
    # Calculate/Load MinHash Signatures with SourMash for MG subseqs
    mg_id = mg_sub_file[0]
    mg_subcontigs = s_utils.get_seqs(mg_sub_file[1])
    mg_headers = tuple(mg_subcontigs.keys())
    kmer_list = [51]
    mh_kmer_recruits_dict = {}
    for kmer in kmer_list:
        print(kmer)
        if ((isfile(o_join(mhr_path, mg_id + '.' + str(kmer) + '.mhr_trimmed_recruits.tsv'))) &
                (force is False)
        ):
            logging.info('MinHash already done, moving on (use --force to re-run)\n')
            mh_max_df = pd.read_csv(o_join(mhr_path, mg_id + '.' + str(kmer) +
                                           '.mhr_trimmed_recruits.tsv'), header=0,
                                    sep='\t'
                                    )
            mh_kmer_recruits_dict[kmer] = mh_max_df

        else:
            build_list, minhash_pass_list = sag_recruit_checker(mhr_path, sag_sub_files, kmer)
            if len(build_list) != 0:
                sag_sig_dict = build_sag_sig_dict(build_list, nthreads, sig_path, kmer)
                # build_mg_lca(mg_id, mg_sub_file, sig_path, nthreads, kmer, checkonly=True)  # make sure SBT exists first
                build_mg_sbt(mg_id, mg_sub_file, sig_path, nthreads, kmer, checkonly=True)  # make sure SBT exists first
                pool = multiprocessing.Pool(processes=nthreads)
                sbt_args = mg_id, mg_sub_file, sig_path, nthreads
                chunk_list = [list(x) for x in np.array_split(np.array(list(sag_sig_dict.keys())),
                                                              nthreads * 10) if len(list(x)) != 0
                              ]  # TODO: might be a better way to chunk up the list?
                logging.info('Built {} Blocks of SAG Signature Sets\n'.format(len(chunk_list)))
                arg_list = []
                for i, sag_id_list in enumerate(chunk_list):
                    sub_sag_sig_dict = {k: sag_sig_dict[k] for k in sag_id_list}
                    arg_list.append([sbt_args, mhr_path, sag_id_list, sub_sag_sig_dict, kmer])
                results = pool.imap_unordered(compare_sag_sbt, arg_list)
                logging.info('Querying {} Signature Blocks against SBT\n'.format(len(chunk_list)))
                logging.info('WARNING: This can be VERY time consuming, '
                             'be patient\n'.format(len(chunk_list))
                             )
                df_cnt = 0
                logging.info('Signatures Queried Against SBT: {}/{}'
                             '\r'.format(df_cnt, len(sag_sig_dict.keys()))
                             )
                for i, search_df in enumerate(results):
                    df_cnt += len(search_df)
                    logging.info('Signatures Queried Against SBT: {}/{}'
                                 '\r'.format(df_cnt, len(sag_sig_dict.keys()))
                                 )
                    minhash_pass_list.extend(search_df)
                logging.info('\n')
                pool.close()
                pool.join()

            if len(minhash_pass_list) > 1:
                minhash_df = pd.concat(minhash_pass_list)
            else:
                minhash_df = minhash_pass_list[0]

            minhash_df['jacc_sim'] = minhash_df['jacc_sim'].astype(float)
            # recruit_list = list(minhash_df['subcontig_id'].loc[minhash_df['jacc_sim'] >= 0.10])
            minhash_recruit_df = minhash_df.copy()  # .loc[minhash_df['subcontig_id'].isin(recruit_list)]
            minhash_recruit_df.to_csv(o_join(mhr_path, mg_id + '.' + str(kmer) +
                                             '.mhr_contig_recruits.tsv'),
                                      sep='\t',
                                      index=False
                                      )
            logging.info('Compiling all MinHash Recruits\n')
            '''
            # Count # of subcontigs recruited to each SAG via sourmash
            mh_cnt_df = minhash_recruit_df.groupby(['sag_id', 'q_contig_id']
                                                   )['q_subcontig_id'].count().reset_index()
            mh_cnt_df.columns = ['sag_id', 'contig_id', 'subcontig_recruits']
            mh_avg_df = minhash_recruit_df.groupby(['sag_id', 'q_contig_id']
                                                   )['jacc_sim'].mean().reset_index()
            mh_avg_df.columns = ['sag_id', 'contig_id', 'jacc_sim_avg']
            mh_max_df = minhash_recruit_df.groupby(['sag_id', 'q_contig_id']
                                                   )['jacc_sim'].max().reset_index()
            mh_max_df.columns = ['sag_id', 'contig_id', 'jacc_sim_max']
            mh_dfs = [mh_cnt_df, mh_avg_df, mh_max_df]
            mh_merge_df = reduce(lambda left, right: pd.merge(left, right,
                                                              on=['sag_id', 'contig_id']),
                                 mh_dfs
                                 )
            '''
            mh_trim_df = minhash_recruit_df[['sag_id', 'q_contig_id', 'jacc_sim']]
            mh_trim_df.columns = ['sag_id', 'contig_id', 'jacc_sim']
            mh_filter_df = mh_trim_df.loc[mh_trim_df['jacc_sim'] >= 0.99]
            '''
            # Build subcontig count for each MG contig
            mg_contig_list = [x.rsplit('_', 1)[0] for x in mg_headers]
            mg_tot_df = pd.DataFrame(zip(mg_contig_list, mg_headers),
                                     columns=['contig_id', 'subcontig_id'])
            mg_tot_cnt_df = mg_tot_df.groupby(['contig_id']).count().reset_index()
            mg_tot_cnt_df.columns = ['contig_id', 'subcontig_total']
            mh_recruit_df = mh_cnt_df.merge(mg_tot_cnt_df, how='left', on='contig_id')
            mh_recruit_df['percent_recruited'] = mh_recruit_df['subcontig_recruits'] / \
                                                 mh_recruit_df['subcontig_total']
            mh_jacc_merge_df = mh_recruit_df.merge(mh_avg_df, how='left', on=['sag_id', 'contig_id'])
            mh_max_merge_df = mh_jacc_merge_df.merge(mh_max_df, how='left', on=['sag_id', 'contig_id'])

            mh_max_list = []
            for sag_id in list(set(minhash_df['sag_id'])):
                sag_max_only_df = mh_max_merge_df.loc[mh_max_merge_df['sag_id'] == sag_id]
                mh_max_df = mg_tot_df[mg_tot_df['contig_id'].isin(list(sag_max_only_df['contig_id']))]
                mh_max_df['sag_id'] = sag_id
                sag_merge_df = mh_max_df.merge(sag_max_only_df, how='left',
                                               on=['contig_id', 'sag_id']
                                               )
                sag_merge_df = sag_merge_df[['sag_id', 'subcontig_id', 'contig_id', 'subcontig_recruits',
                                             'subcontig_total', 'percent_recruited', 'jacc_sim_avg',
                                             'jacc_sim_max'
                                             ]]
                mh_max_list.append(sag_merge_df)
            mh_final_max_df = pd.concat(mh_max_list)
            merge_jacc_df = mh_final_max_df.merge(minhash_df, how='left',
                                                  on=['sag_id', 'subcontig_id', 'contig_id']
                                                  )
            merge_jacc_df.fillna(0, inplace=True)
            minhash_filter_df = merge_jacc_df.loc[(merge_jacc_df['jacc_sim_max'] != 0)]
            minhash_filter_df.to_csv(o_join(mhr_path, mg_id + '.' + str(kmer) + '.mhr_trimmed_recruits.tsv'), sep='\t',
                                     index=False
                                     )
            mh_kmer_recruits_dict[kmer] = minhash_filter_df
            '''
            mh_filter_df.to_csv(o_join(mhr_path, mg_id + '.' + str(kmer) + '.mhr_trimmed_recruits.tsv'), sep='\t',
                                index=False
                                )
            mh_kmer_recruits_dict[kmer] = mh_filter_df

    logging.info('MinHash Recruitment Algorithm Complete\n')
    return mh_kmer_recruits_dict


def build_sag_sig_dict(build_list, nthreads, sig_path, kmer):
    pool = multiprocessing.Pool(processes=nthreads)
    arg_list = []
    for i, sag_rec in enumerate(build_list):
        sag_id, sag_file = sag_rec
        arg_list.append([sag_file, sag_id, sig_path, kmer])
    results = pool.imap_unordered(load_sag_sigs, arg_list)
    sag_sig_dict = {}
    for i, sag_sig_rec in enumerate(results):
        sag_id, sag_sig_list = sag_sig_rec
        logging.info('Loading/Building SAGs Signatures: {}/{}\r'.format(i + 1, len(build_list)))
        sag_sig_dict[sag_id] = sag_sig_list
    logging.info('\n')
    pool.close()
    pool.join()

    return sag_sig_dict


def compare_sag_sbt(p):  # TODO: needs stdout for user monitoring
    sbt_args, mhr_path, sag_id_list, sag_sig_dict, kmer = p
    mg_id, mg_sub_file, sig_path, nthreads = sbt_args
    mg_sbt = build_mg_sbt(mg_id, mg_sub_file, sig_path, nthreads, kmer)
    search_df_list = []
    for i, sag_id in enumerate(sag_id_list):
        sag_sig_list = sag_sig_dict[sag_id]
        search_list = []
        for i, sig in enumerate(sag_sig_list):
            sbt_out = mg_sbt.search(sig, threshold=0.000000000001)
            sbt_out_cont = mg_sbt.search(sig, threshold=0.000000000001, do_containment=True)
            sbt_out.extend(sbt_out_cont)
            # r_subcontig = sig.name()
            # r_contig = r_subcontig.rsplit('_', 1)[0]
            r_contig = sig.name()
            for similarity, t_sig, filename in sbt_out:
                # q_subcontig = t_sig.name()
                # q_contig = q_subcontig.rsplit('_', 1)[0]
                q_contig = t_sig.name()
                # search_list.append([sag_id, r_subcontig, r_contig, q_subcontig,
                #                    q_contig, similarity
                #                    ])
                search_list.append([sag_id, r_contig, q_contig, similarity])
        search_df = pd.DataFrame(search_list, columns=['sag_id', 'r_contig_id', 'q_contig_id',
                                                       'jacc_sim'
                                                       ])
        # search_df = pd.DataFrame(search_list, columns=['sag_id', 'r_subcontig_id', 'r_contig_id',
        #                                               'q_subcontig_id', 'q_contig_id',
        #                                               'jacc_sim'
        #                                               ])
        search_df['jacc_sim'] = search_df['jacc_sim'].astype(float)
        search_df.sort_values(by='jacc_sim', ascending=False, inplace=True)
        search_file = o_join(mhr_path, sag_id + '.' + str(kmer) + '.mhr_recruits.tsv')
        search_df.to_csv(search_file, sep='\t',
                         index=False
                         )
        search_df_list.append(search_df)
    return search_df_list


def compare_sag_lca(p):  # TODO: needs stdout for user monitoring
    lca_args, mhr_path, sag_id_list, sag_sig_dict, kmer = p
    mg_id, mg_sub_file, sig_path, nthreads = lca_args
    mg_lca = build_mg_lca(mg_id, mg_sub_file, sig_path, nthreads, kmer)
    search_df_list = []
    for i, sag_id in enumerate(sag_id_list):
        sag_sig_list = sag_sig_dict[sag_id]
        search_list = []
        for i, sig in enumerate(sag_sig_list):
            lca_out = mg_lca.search(sig, threshold=0.000000000001)
            for similarity, t_sig, filename in lca_out:
                q_subcontig = t_sig.name()
                q_contig = q_subcontig.rsplit('_', 1)[0]
                search_list.append([sag_id, q_subcontig, q_contig, similarity])
        search_df = pd.DataFrame(search_list, columns=['sag_id', 'subcontig_id', 'contig_id',
                                                       'jacc_sim'
                                                       ])
        search_df['jacc_sim'] = search_df['jacc_sim'].astype(float)
        search_df.sort_values(by='jacc_sim', ascending=False, inplace=True)
        search_df.drop_duplicates(subset='subcontig_id', inplace=True)
        search_file = o_join(mhr_path, sag_id + '.' + str(kmer) + '.mhr_recruits.tsv')
        search_df.to_csv(search_file, sep='\t',
                         index=False
                         )
        search_df_list.append(search_df)
    return search_df_list


def build_mg_lca(mg_id, mg_sub_file, sig_path, nthreads, kmer, checkonly=False):
    mg_lca_file = o_join(sig_path, mg_id + '.' + str(kmer) + '.lca.json.gz')
    if isfile(mg_lca_file):
        if checkonly is True:
            logging.info('%s LCA DB Exists\n' % mg_id)
            mg_lca_db = True
        else:
            mg_lca_db = sourmash.load_file_as_index(mg_lca_file)
    else:
        logging.info('Building %s LCA Database\n' % mg_id)
        mg_sig_list = load_mg_sigs(mg_id, mg_sub_file, nthreads, sig_path, kmer)
        db = LCA_Database(ksize=kmer, scaled=100, moltype='DNA')
        pool = multiprocessing.Pool(processes=nthreads)
        for i, sig in enumerate(mg_sig_list, 1):
            hashes_inserted = db.insert(sig)
            logging.info('Inserting hashes into LCA DB: {}/{}\r'.format(i, len(mg_sig_list)))
        logging.info('\n')
        db.save(mg_lca_file)
        pool.close()
        pool.join()
        mg_lca_db = None

    return mg_lca_db


def build_mg_sbt(mg_id, mg_sub_file, sig_path, nthreads, kmer, checkonly=False):
    mg_sbt_file = o_join(sig_path, mg_id + '.' + str(kmer) + '.sbt.zip')
    if isfile(mg_sbt_file):
        if checkonly is True:
            logging.info('%s Sequence Bloom Tree Exists\n' % mg_id)
            mg_sbt_tree = True
        else:
            mg_sbt_tree = sourmash.load_sbt_index(mg_sbt_file)
    else:
        logging.info('Building %s Sequence Bloom Tree\n' % mg_id)  # TODO: perhaps multiple smaller SBTs would be better
        mg_sig_list = load_mg_sigs(mg_id, mg_sub_file, nthreads, sig_path, kmer)
        mg_sbt_tree = sourmash.create_sbt_index()
        pool = multiprocessing.Pool(processes=nthreads)
        results = pool.imap_unordered(build_leaf, mg_sig_list)
        leaf_list = []
        for i, leaf in enumerate(results, 1):
            logging.info('Building leaves for SBT: {}/{}\r'.format(i, len(mg_sig_list)))
            leaf_list.append(leaf)
        leaf_list = tuple(leaf_list)
        logging.info('\n')
        for i, lef in enumerate(leaf_list, 1):
            logging.info('Adding leaves to tree: {}/{}\r'.format(i, len(leaf_list)))
            mg_sbt_tree.add_node(lef)
        logging.info('\n')
        mg_sbt_tree.save(mg_sbt_file)
        pool.close()
        pool.join()
        mg_sbt_tree = None  # sourmash.load_sbt_index(mg_sbt_file)

    return mg_sbt_tree


def build_leaf(sig):
    leaf = SigLeaf(sig.md5sum(), sig)
    return leaf


def load_mg_sigs(mg_id, mg_sub_file, nthreads, sig_path, kmer):
    if isfile(o_join(sig_path, mg_id + '.' + str(kmer) + '.metaG.sig')):
        logging.info('Loading %s Signatures\n' % mg_id)
        mg_sig_list = tuple(sourmash.signature.load_signatures(o_join(sig_path, mg_id + \
                                                                      '.' + str(kmer) + '.metaG.sig')
                                                               ))
    else:
        logging.info('Loading subcontigs for %s\n' % mg_id)
        mg_subcontigs = s_utils.get_seqs(mg_sub_file[1])
        mg_sig_list = build_mg_sigs(mg_id, mg_subcontigs, nthreads, sig_path, kmer)
    return mg_sig_list


def load_sag_sigs(p):
    sag_file, sag_id, sig_path, kmer = p
    if isfile(o_join(sig_path, sag_id + '.' + str(kmer) + '.SAG.sig')):
        sag_sig_list = tuple(sourmash.signature.load_signatures(o_join(sig_path,
                                                                       sag_id + '.' + str(kmer) + '.SAG.sig')
                                                                ))
    else:
        sag_sig_list = build_sag_sigs(sag_file, sag_id, sig_path, kmer)
    return sag_id, sag_sig_list


def build_sag_sigs(sag_file, sag_id, sig_path, kmer):
    sag_subcontigs = s_utils.get_seqs(sag_file)
    sag_headers = tuple(sag_subcontigs.keys())
    sag_sig_list = []
    for i, sag_head in enumerate(sag_headers):
        sag_sig = build_signature([sag_head, str(sag_subcontigs[sag_head].seq), kmer])
        sag_sig_list.append(sag_sig)
    with open(o_join(sig_path, sag_id + '.' + str(kmer) + '.SAG.sig'), 'w') as sag_out:
        sourmash.signature.save_signatures(sag_sig_list, fp=sag_out)
    sag_sig_list = tuple(sag_sig_list)
    return sag_sig_list


def build_mg_sigs(mg_id, mg_subcontigs, nthreads, sig_path, kmer):
    mg_headers = mg_subcontigs.keys()
    arg_list = []
    for i, mg_head in enumerate(mg_headers):
        arg_list.append([mg_head, str(mg_subcontigs[mg_head].seq), kmer])
    pool = multiprocessing.Pool(processes=nthreads)
    results = pool.imap_unordered(build_signature, arg_list)
    mg_sig_list = []
    for i, mg_sig in enumerate(results, 1):
        logging.info('Building MinHash Signatures for {}: {}/{} done\r'.format(mg_id, i, len(arg_list)))
        mg_sig_list.append(mg_sig)
    logging.info('\n')
    pool.close()
    pool.join()
    with open(o_join(sig_path, mg_id + '.' + str(kmer) + '.metaG.sig'), 'w') as mg_out:
        sourmash.signature.save_signatures(mg_sig_list, fp=mg_out)
    mg_sig_list = tuple(sourmash.signature.load_signatures(o_join(sig_path, mg_id + '.' + str(kmer) + '.metaG.sig')))

    return mg_sig_list


def sag_recruit_checker(mhr_path, sag_sub_files, kmer):
    build_list = []
    minhash_pass_list = []
    l = 0
    b = 0
    for i, sag_rec in enumerate(sag_sub_files):
        sag_id, sag_file = sag_rec
        mh_file = o_join(mhr_path, sag_id + '.' + str(kmer) + '.mhr_recruits.tsv')
        if isfile(mh_file):
            filesize = getsize(mh_file)
        else:
            filesize = 0
        if filesize != 0:
            pass_df = pd.read_csv(mh_file, header=0, sep='\t')
            minhash_pass_list.append(pass_df)
            l += 1
        else:
            build_list.append(sag_rec)
            b += 1
        logging.info('Checking for previously completed SAGs: {}/{} done\r'.format(l, b))
    logging.info('\n')
    return build_list, minhash_pass_list


def build_signature(p):
    header, seq, kmer = p
    mg_minhash = sourmash.MinHash(ksize=kmer, scaled=1000, n=0)
    mg_minhash.add_sequence(str(seq), force=True)
    mg_sig = sourmash.SourmashSignature(mg_minhash, name=header)

    return mg_sig


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='uses MinHash Signatures to recruit metaG reads to SAGs')
    parser.add_argument("-s", "--sag", required=True, dest="sag_path",
                        help="Path to reference SAG FASTA file or directory "
                             "containing only FASTA files."
                        )
    parser.add_argument("-m", "--metag", required=True, dest="mg_file",
                        help="Path to a metagenome assembly [FASTA format only]."
                        )
    parser.add_argument("-o", "--output-dir", required=True, dest="save_path",
                        help="Path to directory for all outputs."
                        )
    parser.add_argument("--max_contig_len", required=False, default=10000,
                        dest="max_contig_len",
                        help="Max subcontig length in basepairs [10000]."
                        )
    parser.add_argument("--overlap_len", required=False, default=2000,
                        dest="overlap_len",
                        help="subcontig overlap in basepairs [2000]."
                        )
    parser.add_argument(
        '--threads',
        help='number of threads to use', required=False, default='2'
    )
    parser.add_argument("--force", required=False, default=False,
                        help="Force SABer to run even if final recruits files exist [False]"
                        )
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Prints a more verbose runtime log"
                        )

    args = parser.parse_args()
    # set args
    save_path = args.save_path
    sag_path = args.sag_path
    mg_file = args.mg_file
    max_contig_len = args.max_contig_len
    overlap_len = args.overlap_len
    nthreads = int(args.threads)
    force = args.force

    # Build save dir structure
    save_dirs_dict = s_utils.check_out_dirs(save_path)
    # Find the SAGs!
    sag_list = s_utils.get_SAGs(sag_path)

    # Build subcontiges for SAGs and MG
    sag_sub_files = s_utils.build_subcontigs('SAGs', sag_list,
                                             save_dirs_dict['subcontigs'],
                                             max_contig_len,
                                             overlap_len
                                             )

    mg_sub_file = s_utils.build_subcontigs('Metagenomes', [mg_file],
                                           save_dirs_dict['subcontigs'],
                                           max_contig_len,
                                           overlap_len
                                           )
    # setup logging
    logging.StreamHandler.terminator = ""
    logging.getLogger().setLevel(logging.INFO)
    s_log.prep_logging("minhash_log.txt", args.verbose)

    run_minhash_recruiter(save_dirs_dict['signatures'], save_dirs_dict['minhash_recruits'],
                          sag_sub_files, mg_sub_file, nthreads, force
                          )
