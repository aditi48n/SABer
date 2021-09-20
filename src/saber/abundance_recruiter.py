import argparse
import logging
from os.path import isfile, basename, getsize
from os.path import join as o_join
from subprocess import Popen

import pandas as pd
from sklearn import svm

import utilities as s_utils

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None
import multiprocessing
from sklearn.preprocessing import StandardScaler
import tetranuc_recruiter as tra
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest


def runAbundRecruiter(subcontig_path, abr_path, mg_sub_file, mg_raw_file_list,
                      minhash_dict, nu, gamma, nthreads, force
                      ):
    logging.info('Starting Abundance Recruitment\n')
    mg_id = mg_sub_file[0]
    if ((isfile(o_join(abr_path, mg_id + '.abr_trimmed_recruits.tsv'))) &
            (force is False)
    ):
        logging.info('Loading Abundance matrix for %s\n' % mg_id)
        covm_df = pd.read_csv(o_join(abr_path, mg_id + '.abr_trimmed_recruits.tsv'), header=0,
                              sep='\t'
                              )
    else:
        logging.info('Building %s abundance table\n' % mg_id)
        mg_sub_path = o_join(subcontig_path, mg_id + '.subcontigs.fasta')
        # Process raw metagenomes to calculate abundances
        mg_covm_out = procMetaGs(abr_path, mg_id, mg_sub_path, mg_raw_file_list,
                                 subcontig_path, nthreads
                                 )
        # Recruit subcontigs using OC-SVM
        # minhash_df['jacc_sim'] = minhash_df['jacc_sim'].astype(float)
        mh_recruits_df = minhash_dict[201]
        # Filter out contigs that didn't meet MinHash recruit standards
        # mh_perfect_df = mh_recruits_df.loc[mh_recruits_df['jacc_sim_max'] >= 0.90]
        logging.info("Starting one-class SVM analysis\n")
        covm_df = abund_recruiter(abr_path, mg_covm_out, mh_recruits_df, mh_recruits_df,
                                  nu, gamma, nthreads
                                  )
        covm_df.to_csv(o_join(abr_path, mg_id + '.abr_trimmed_recruits.tsv'),
                       sep='\t', index=False
                       )

    return covm_df


def abund_recruiter(abr_path, mg_covm_out, minhash_all_df, minhash_df, nu, gamma, nthreads):
    covm_pass_dfs = []
    pool = multiprocessing.Pool(processes=nthreads)
    arg_list = []
    # Prep MinHash
    # minhash_df.sort_values(by='jacc_sim', ascending=False, inplace=True)
    minhash_dedup_df = minhash_df[['sag_id', 'contig_id', 'jacc_sim']]
    mh_recruit_dict = tra.build_uniq_dict(minhash_dedup_df, 'sag_id', nthreads,
                                          'MinHash Recruits')  # TODO: this might not need multithreading
    for i, sag_id in enumerate(list(mh_recruit_dict.keys()), 1):
        logging.info('\rPrepping for OCSVM: {}/{}'.format(i, len(mh_recruit_dict.keys())))
        if isfile(o_join(abr_path, sag_id + '.abr_recruits.tsv')):
            final_pass_df = pd.read_csv(o_join(abr_path, sag_id + '.abr_recruits.tsv'),
                                        header=None,
                                        names=['sag_id', 'subcontig_id', 'contig_id'],
                                        sep='\t'
                                        )
            dedupped_pass_df = final_pass_df[['sag_id', 'contig_id']].drop_duplicates(
                subset=['sag_id', 'contig_id']
            )
            covm_pass_dfs.append(dedupped_pass_df)

        else:
            mh_sub_df = mh_recruit_dict[sag_id]
            mg_sub_df = minhash_all_df.loc[minhash_all_df['sag_id'] == sag_id]
            arg_list.append([abr_path, sag_id, mh_sub_df, mg_covm_out, mg_sub_df, nu, gamma])
    logging.info('\n')
    logging.info("{} already complete, {} to run\n".format(len(covm_pass_dfs), len(arg_list)))
    results = pool.imap_unordered(recruitSubs, arg_list)
    for i, output in enumerate(results, 1):
        logging.info('\rRecruiting with Abundance Model: {}/{}'.format(i, len(arg_list)))
        covm_pass_dfs.append(output)
    logging.info('\n')
    pool.close()
    pool.join()
    covm_df = pd.concat(covm_pass_dfs)
    return covm_df


def procMetaGs(abr_path, mg_id, mg_sub_path, mg_raw_file_list, subcontig_path, nthreads):
    # Build BWA index
    # buildBWAindex(abr_path, mg_id, mg_sub_path)
    # Process each raw metagenome
    with open(mg_raw_file_list, 'r') as raw_fa_in:
        raw_data = raw_fa_in.readlines()
    sorted_bam_list = []
    for line in raw_data:
        raw_file_list = line.strip('\n').split('\t')
        # Run BWA mem
        pe_id, mg_sam_out = runBWAmem(abr_path, subcontig_path, mg_id, raw_file_list,
                                      nthreads
                                      )
        # Build/sorted .bam files
        mg_sort_out = runSamTools(abr_path, pe_id, nthreads, mg_id, mg_sam_out)
        sorted_bam_list.append(mg_sort_out)
    logging.info('\n')
    mg_covm_out = runCovM(abr_path, mg_id, nthreads, sorted_bam_list)

    return mg_covm_out


def buildBWAindex(abr_path, mg_id, mg_sub_path):
    index_ext_list = ['amb', 'ann', 'bwt', 'pac', 'sa']
    check_ind_list = ['.'.join([mg_sub_path, x]) for x in index_ext_list]
    if False in (isfile(f) for f in check_ind_list):
        # Use BWA to build an index for metagenome assembly
        logging.info('Creating index with BWA\n')
        mg_sub_fa = s_utils.get_seqs(mg_sub_path)
        base_count = mg_sub_fa.size
        bwa_cmd = ['bwa', 'index', '-b', str(int(base_count / 8)), mg_sub_path]
        with open(o_join(abr_path, mg_id + '.stdout.txt'), 'w') as stdout_file:
            with open(o_join(abr_path, mg_id + '.stderr.txt'), 'w') as stderr_file:
                run_bwa = Popen(bwa_cmd, stdout=stdout_file,
                                stderr=stderr_file
                                )
                run_bwa.communicate()

    return


def runBWAmem(abr_path, subcontig_path, mg_id, raw_file_list, nthreads):
    if len(raw_file_list) == 2:
        logging.info('Raw reads in FWD and REV file...\n')
        pe1 = raw_file_list[0]
        pe2 = raw_file_list[1]
        mem_cmd = ['minimap2', '-ax', 'sr', '-t', str(nthreads),
                   o_join(subcontig_path, mg_id + '.subcontigs.fasta'), pe1, pe2
                   ]  # TODO: add support for specifying number of threads
    else:  # if the fastq is interleaved
        logging.info('Raw reads in interleaved file...\n')
        pe1 = raw_file_list[0]
        mem_cmd = ['minimap2', '-ax', 'sr', '-t', str(nthreads),
                   o_join(subcontig_path, mg_id + '.subcontigs.fasta'), pe1
                   ]  # TODO: how to get install path for executables?
    pe_basename = basename(pe1)
    pe_id = pe_basename.split('.')[0]
    # BWA sam file exists?
    mg_sam_out = o_join(abr_path, pe_id + '.sam')
    if isfile(mg_sam_out) == False:
        logging.info('Running minimap2-sr on %s\n' % pe_id)
        with open(mg_sam_out, 'w') as sam_file:
            with open(o_join(abr_path, pe_id + '.stderr.txt'), 'w') as stderr_file:
                run_mem = Popen(mem_cmd, stdout=sam_file, stderr=stderr_file)
                run_mem.communicate()

    return pe_id, mg_sam_out


def runSamTools(abr_path, pe_id, nthreads, mg_id, mg_sam_out):
    mg_bam_out = o_join(abr_path, pe_id + '.bam')
    if isfile(mg_bam_out) == False:
        logging.info('Converting SAM to BAM with SamTools\n')
        bam_cmd = ['samtools', 'view', '-S', '-b', '-@', str(nthreads), mg_sam_out]
        with open(mg_bam_out, 'w') as bam_file:
            with open(o_join(abr_path, mg_id + '.stderr.txt'), 'w') as stderr_file:
                run_bam = Popen(bam_cmd, stdout=bam_file, stderr=stderr_file)
                run_bam.communicate()
    # sort bam file
    mg_sort_out = o_join(abr_path, pe_id + '.sorted.bam')
    if isfile(mg_sort_out) == False:
        logging.info('Sort BAM with SamTools\n')
        sort_cmd = ['samtools', 'sort', '-@', str(nthreads), mg_bam_out, '-o', mg_sort_out]
        with open(o_join(abr_path, mg_id + '.stderr.txt'), 'w') as stderr_file:
            run_sort = Popen(sort_cmd, stderr=stderr_file)
            run_sort.communicate()

    return mg_sort_out


def runCovM(abr_path, mg_id, nthreads, sorted_bam_list):
    # run coverm on sorted bams
    mg_covm_out = o_join(abr_path, mg_id + '.metabat.tsv')
    mg_covm_std = o_join(abr_path, mg_id + '.covM.scaled.tsv')
    try:  # if file exists but is empty
        covm_size = getsize(mg_covm_std)
    except:  # if file doesn't exist
        covm_size = -1
    if covm_size <= 0:
        logging.info('Calculate mean abundance and variance with CoverM\n')
        covm_cmd = ['coverm', 'contig', '-t', str(nthreads), '-m', 'metabat', '-b'
                    ]
        covm_cmd.extend(sorted_bam_list)
        with open(mg_covm_out, 'w') as covm_file:
            with open(o_join(abr_path, mg_id + '.stderr.txt'), 'w') as stderr_file:
                run_covm = Popen(covm_cmd, stdout=covm_file, stderr=stderr_file)
                run_covm.communicate()
        mg_covm_df = pd.read_csv(mg_covm_out, header=0, sep='\t', index_col=['contigName'])
        mg_covm_df.drop(columns=['contigLen', 'totalAvgDepth'], inplace=True)
        scale = StandardScaler().fit(mg_covm_df.values)
        scaled_data = scale.transform(mg_covm_df.values)
        std_merge_df = pd.DataFrame(scaled_data, index=mg_covm_df.index).reset_index()
        std_merge_df.to_csv(mg_covm_std, header=True, sep='\t', index=False)

    return mg_covm_std


def recruitSubs(p):
    abr_path, sag_id, minhash_sag_df, mg_covm_out, mg_sub_df, nu, gamma = p
    abr_recruit_file = o_join(abr_path, sag_id + '.abr_recruits.tsv')
    if len(minhash_sag_df['sag_id']) != 0:
        mg_covm_df = pd.read_csv(mg_covm_out, header=0, sep='\t', index_col=['contigName'])
        mg_covm_df['contig_id'] = [x.rsplit('_', 1)[0] for x in mg_covm_df.index.values]
        mg_covm_filter_df = mg_covm_df.copy()  # mg_covm_df.loc[mg_covm_df['contig_id'].isin(list(mg_sub_df['contig_id']))]
        recruit_contigs_df = mg_covm_filter_df.loc[mg_covm_filter_df['contig_id'].isin(
            list(minhash_sag_df['contig_id']))
        ]
        mg_covm_filter_df.drop(columns=['contig_id'], inplace=True)
        recruit_contigs_df.drop(columns=['contig_id'], inplace=True)

        # mg_covm_filter_df = mg_covm_df.loc[mg_covm_df.index.isin(list(mg_sub_df['subcontig_id']))]
        # recruit_contigs_df = mg_covm_df.loc[mg_covm_df.index.isin(
        #    list(minhash_sag_df['subcontig_id']))
        # ]
        mh_recruit_list = [[sag_id, x, x.rsplit('_', 1)[0]] for x in recruit_contigs_df.index]
        mh_recruit_df = pd.DataFrame(mh_recruit_list, columns=['sag_id', 'subcontig_id', 'contig_id'])
        kmeans_pass_list, kclusters_df = runKMEANS(recruit_contigs_df, sag_id, mg_covm_filter_df)
        # print('\n')
        # print(kclusters_df.head())
        # print(kclusters_df.loc[kclusters_df['subcontig_id'].isin(
        #                                            list(minhash_sag_df['subcontig_id']))
        #      ].head())
        # print(len(kclusters_df['kmeans_clust'].unique()))
        kmeans_pass_df = pd.DataFrame(kmeans_pass_list,
                                      columns=['sag_id', 'subcontig_id', 'contig_id']
                                      )
        nonrecruit_kmeans_df = mg_covm_filter_df.loc[mg_covm_filter_df.index.isin(
            kmeans_pass_df['subcontig_id']
        )]
        final_pass_list = runOCSVM(recruit_contigs_df, nonrecruit_kmeans_df, sag_id, nu, gamma)
        # final_pass_list = runIFOREST(recruit_contigs_df, nonrecruit_kmeans_df, sag_id)
        final_pass_df = pd.DataFrame(final_pass_list,
                                     columns=['sag_id', 'subcontig_id', 'contig_id']
                                     )
        mh_abr_df = pd.concat([final_pass_df, mh_recruit_df])
        mh_abr_df.to_csv(abr_recruit_file, header=False, index=False, sep='\t')

    else:
        mh_abr_df = pd.DataFrame([], columns=['sag_id', 'subcontig_id', 'contig_id'])
    dedupped_pass_df = mh_abr_df[['sag_id', 'contig_id']].drop_duplicates(
        subset=['sag_id', 'contig_id']
    )
    return dedupped_pass_df


def runKMEANS(recruit_contigs_df, sag_id, std_merge_df):
    temp_cat_df = std_merge_df.copy()
    last_len = 0
    while temp_cat_df.shape[0] != last_len:
        last_len = temp_cat_df.shape[0]
        clusters = 10 if last_len >= 10 else last_len
        kmeans = MiniBatchKMeans(n_clusters=clusters, random_state=42).fit(temp_cat_df.values)
        clust_labels = kmeans.labels_
        clust_df = pd.DataFrame(zip(temp_cat_df.index.values, clust_labels),
                                columns=['subcontig_id', 'kmeans_clust']
                                )
        recruit_clust_df = clust_df.loc[clust_df['subcontig_id'].isin(list(recruit_contigs_df.index))]
        subset_clust_df = clust_df.loc[clust_df['kmeans_clust'].isin(
            list(recruit_clust_df['kmeans_clust'].unique())
        )]
        subset_clust_df['kmeans_pred'] = 1
        temp_cat_df = temp_cat_df.loc[temp_cat_df.index.isin(list(subset_clust_df['subcontig_id']))]
    cat_clust_df = subset_clust_df.copy()  # pd.concat(block_list)
    std_id_df = pd.DataFrame(std_merge_df.index.values, columns=['subcontig_id'])
    std_id_df['contig_id'] = [x.rsplit('_', 1)[0] for x in std_id_df['subcontig_id']]
    cat_clust_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cat_clust_df['subcontig_id']]
    sub_std_df = std_id_df.loc[std_id_df['contig_id'].isin(list(cat_clust_df['contig_id']))]
    std_clust_df = sub_std_df.merge(cat_clust_df, on=['subcontig_id', 'contig_id'], how='outer')
    std_clust_df.fillna(-1, inplace=True)
    pred_df = std_clust_df[['subcontig_id', 'contig_id', 'kmeans_pred']]
    val_perc = pred_df.groupby('contig_id')['kmeans_pred'].value_counts(normalize=True).reset_index(name='percent')
    pos_perc = val_perc.loc[val_perc['kmeans_pred'] == 1]
    major_df = pos_perc.copy()  # = pos_perc.loc[pos_perc['percent'] >= 0.95]
    major_pred_df = pred_df.loc[pred_df['contig_id'].isin(major_df['contig_id'])]
    std_clust_pred_df = std_clust_df.merge(major_pred_df, on=['subcontig_id', 'contig_id'],
                                           how='left'
                                           )
    filter_clust_pred_df = std_clust_pred_df.loc[std_clust_pred_df['kmeans_pred_x'] == 1]
    kmeans_pass_list = []
    for md_nm in major_pred_df['subcontig_id']:
        kmeans_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])
    return kmeans_pass_list, filter_clust_pred_df


def runOCSVM(sag_df, mg_df, sag_id, nu, gamma):
    # fit OCSVM
    clf = svm.OneClassSVM(nu=nu, gamma=gamma)
    clf.fit(sag_df.values)
    mg_pred = clf.predict(mg_df.values)
    contig_id_list = [x.rsplit('_', 1)[0] for x in mg_df.index.values]
    pred_df = pd.DataFrame(zip(mg_df.index.values, contig_id_list, mg_pred),
                           columns=['subcontig_id', 'contig_id', 'ocsvm_pred']
                           )
    val_perc = pred_df.groupby('contig_id')['ocsvm_pred'].value_counts(
        normalize=True).reset_index(name='percent')
    pos_perc = val_perc.loc[val_perc['ocsvm_pred'] == 1]
    major_df = pos_perc.copy()  # .loc[pos_perc['percent'] >= 0.01]
    major_pred_df = pred_df.loc[pred_df['contig_id'].isin(major_df['contig_id'])]
    svm_pass_list = []
    for md_nm in major_pred_df['subcontig_id']:
        svm_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])
    return svm_pass_list


def iqr_bounds(scores, k=1.5):
    q1 = scores.quantile(0.25)
    q3 = scores.quantile(0.75)
    iqr = q3 - q1
    lower_bound = (q1 - k * iqr)
    upper_bound = (q3 + k * iqr)
    return lower_bound, upper_bound


def runIFOREST(sag_df, mg_df, sag_id):
    # fit IsoForest
    clf = IsolationForest(n_estimators=1000, random_state=42)
    clf.fit(sag_df.values)
    sag_pred = clf.predict(sag_df.values)
    sag_scores = clf.score_samples(sag_df.values)
    sag_dfunct = clf.decision_function(sag_df.values)
    sag_anomaly = sag_scores / sag_dfunct
    sag_pred_df = pd.DataFrame(data=sag_pred, index=sag_df.index.values,
                               columns=['anomaly'])
    sag_pred_df.loc[sag_pred_df['anomaly'] == 1, 'anomaly'] = 0
    sag_pred_df.loc[sag_pred_df['anomaly'] == -1, 'anomaly'] = 1
    sag_pred_df['scores'] = sag_scores
    sag_pred_df['decision_function'] = sag_dfunct
    sag_pred_df['normed_anomaly'] = sag_anomaly
    lower_bound, upper_bound = iqr_bounds(sag_pred_df['normed_anomaly'], k=0.5)

    mg_pred = clf.predict(mg_df.values)
    mg_scores = clf.score_samples(mg_df.values)
    mg_dfunct = clf.decision_function(mg_df.values)
    mg_anomaly = mg_scores / mg_dfunct
    mg_pred_df = pd.DataFrame(data=mg_pred, index=mg_df.index.values,
                              columns=['anomaly'])
    mg_pred_df.loc[mg_pred_df['anomaly'] == 1, 'anomaly'] = 0
    mg_pred_df.loc[mg_pred_df['anomaly'] == -1, 'anomaly'] = 1
    mg_pred_df['scores'] = mg_scores
    mg_pred_df['decision_function'] = mg_dfunct
    mg_pred_df['normed_anomaly'] = mg_anomaly
    mg_pred_df['iqr_anomaly'] = 0
    mg_pred_df['iqr_anomaly'] = (mg_pred_df['normed_anomaly'] < lower_bound) | \
                                (mg_pred_df['normed_anomaly'] > upper_bound)
    mg_pred_df['iqr_anomaly'] = mg_pred_df['iqr_anomaly'].astype(int)
    # iso_pass_df = mg_pred_df.loc[mg_pred_df['iqr_anomaly'] != 1]
    mg_pred_df['contig_id'] = [x.rsplit('_', 1)[0] for x in mg_pred_df.index.values]

    val_perc = mg_pred_df.groupby('contig_id')['iqr_anomaly'].value_counts(
        normalize=True).reset_index(name='percent')
    pos_perc = val_perc.loc[val_perc['iqr_anomaly'] == 1]
    major_df = pos_perc.loc[pos_perc['percent'] >= 0.51]
    major_pred_df = mg_pred_df.loc[mg_pred_df['contig_id'].isin(major_df['contig_id'])]
    iso_pass_list = []
    for md_nm in major_pred_df.index.values:
        iso_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])
    return iso_pass_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='uses metabat normalized abundance to recruit metaG reads to SAGs')
    parser.add_argument(
        '--abr_path', help='path to abundance output directory',
        required=True
    )
    parser.add_argument(
        '--sub_path',
        help='path to SAG subcontigs file(s)', required=True
    )
    parser.add_argument(
        '--mg_sub_file',
        help='path to metagenome subcontigs file', required=True
    )
    parser.add_argument(
        "-l", "--metaraw", required=True, dest="mg_raw_file_list",
        help="Text file containing paths to raw FASTQ files for samples. "
             "One file per line, supports interleaved and separate PE reads. "
             "For separate PE files, both file paths on one line sep by [tab]."
    )
    parser.add_argument(
        '--minh_df',
        help='path to output dataframe from abundance recruiter', required=True
    )
    parser.add_argument(
        '--nu_val',
        help='set value for NU, used by OC-SVM', required=True,
        default='0.3'
    )
    parser.add_argument(
        '--gamma_val',
        help='set value for NU, used by OC-SVM', required=True,
        default='10'
    )
    parser.add_argument(
        '--threads',
        help='number of threads to use [1]', required=True,
        default='1'
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Prints a more verbose runtime log"
                        )
    args = parser.parse_args()
    # set args
    abr_path = args.abr_path
    subcontig_path = args.sub_path
    mg_sub_file = args.mg_sub_file
    mg_raw_file_list = args.mg_raw_file_list
    minhash_recruit_file = args.minh_df
    nu = float(args.nu_val)
    gamma = int(args.gamma_val)
    nthreads = int(args.threads)

    s_log.prep_logging("abund_log.txt", args.verbose)
    mg_id = basename(mg_sub_file).rsplit('.', 2)[0]
    minhash_recruit_df = pd.read_csv(minhash_recruit_file, header=0, sep='\t')

    runAbundRecruiter(subcontig_path, abr_path, [mg_id, mg_sub_file], mg_raw_file_list,
                      minhash_recruit_df, nu, gamma, nthreads, False
                      )
