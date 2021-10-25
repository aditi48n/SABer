import logging
from os.path import isfile, basename, getsize
from os.path import join as o_join
from subprocess import Popen

import pandas as pd

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import StandardScaler
from samsum import commands as samsum_cmd
import sys
import pysam
from tqdm import tqdm
import utilities as s_utils


def runAbundRecruiter(subcontig_path, abr_path, mg_sub_file, mg_raw_file_list,
                      nthreads
                      ):
    logging.info('Starting Abundance Data Transformation\n')
    mg_id = mg_sub_file[0]

    if isfile(o_join(abr_path, mg_id + '.coverage.scaled.tsv')):
        logging.info('Loading Abundance matrix for %s\n' % mg_id)
        mg_covm_out = o_join(abr_path, mg_id + '.coverage.scaled.tsv')
    else:
        logging.info('Building %s abundance matrix\n' % mg_id)
        mg_sub_path = o_join(subcontig_path, mg_id + '.subcontigs.fasta')
        # Process raw metagenomes to calculate abundances
        mg_covm_out = procMetaGs(abr_path, mg_id, mg_raw_file_list,
                                 subcontig_path, nthreads
                                 )
    # Clean up the directory
    '''
    logging.info('Cleaning up intermediate files...\n')
    for s in ["*.sam", "*.bam", "*.stderr.txt", "*.stdout.txt", "*.ss_coverage.tsv",
              "*.basecov.txt", "*.constats.txt", "*.covhist.txt", "*.rpkm.txt",
              "ref"]:
        s_utils.runCleaner(abr_path, s)
    '''
    return mg_covm_out


def procMetaGs(abr_path, mg_id, mg_raw_file_list, subcontig_path, nthreads):
    # Process each raw metagenome
    with open(mg_raw_file_list, 'r') as raw_fa_in:
        raw_data = raw_fa_in.readlines()
    sam_list = []
    sorted_bam_list = []
    for line in raw_data:
        raw_file_list = line.strip('\n').split('\t')
        # mg_covm_std = runBBtools(abr_path, subcontig_path, mg_id,
        #                                      raw_file_list, nthreads
        #                                      )
        pe_id, mg_sam_out = runMiniMap2(abr_path, subcontig_path, mg_id, raw_file_list,
                                        nthreads
                                        )
        sam_list.append(mg_sam_out)
        # Build/sorted .bam files
        mg_sort_out = runSamTools(abr_path, pe_id, nthreads, mg_id, mg_sam_out)
        sorted_bam_list.append(mg_sort_out)
    logging.info('\n')
    mg_covm_out = runMBAcov(abr_path, mg_id, sorted_bam_list)
    # mg_covm_out = runCovM(abr_path, mg_id, nthreads, sorted_bam_list)
    # mg_covm_out = runSAMSAM(abr_path, subcontig_path, mg_id, sam_list, nthreads)
    # mg_covm_out = runPySAM(abr_path, subcontig_path, mg_id, sorted_bam_list, nthreads)
    return mg_covm_out


def runMiniMap2(abr_path, subcontig_path, mg_id, raw_file_list, nthreads):
    pe1 = raw_file_list[0]
    if isfile(pe1) == True:
        pe_basename = basename(pe1)
        pe_id = pe_basename.split('.')[0]
        mg_sam_out = o_join(abr_path, pe_id + '.sam')
        if len(raw_file_list) == 2:
            logging.info('Raw reads in FWD and REV file...\n')
            pe2 = raw_file_list[1]
            mem_cmd = ['minimap2', '-ax', 'sr', '-t', str(nthreads), '-o', mg_sam_out,
                       o_join(subcontig_path, mg_id + '.subcontigs.fasta'), pe1, pe2
                       ]
        else:  # if the fastq is interleaved
            logging.info('Raw reads in interleaved file...\n')
            mem_cmd = ['minimap2', '-ax', 'sr', '-t', str(nthreads), '-o', mg_sam_out,
                       o_join(subcontig_path, mg_id + '.subcontigs.fasta'), pe1
                       ]

        if isfile(mg_sam_out) == False:
            logging.info('Running minimap2-sr on %s\n' % pe_id)
            with open(mg_sam_out, 'w') as sam_file:
                with open(o_join(abr_path, pe_id + '.stderr.txt'), 'w') as stderr_file:
                    with open(o_join(abr_path, pe_id + '.stdout.txt'), 'w') as stdout_file:
                        run_mem = Popen(mem_cmd, stdout=stdout_file, stderr=stderr_file)
                        run_mem.communicate()
        else:
            print('SAM file already exists, skipping alignment...')
    else:
        print('Raw FASTQ file(s) are not where you said they were...')
        sys.exit()  # TODO: replace this quick-fix with a real exception

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


def runMBAcov(abr_path, mg_id, sorted_bam_list):
    # run mba on sorted bams
    mg_mba_out = o_join(abr_path, mg_id + '.mbacov.tsv')
    mg_mba_std = o_join(abr_path, mg_id + '.coverage.scaled.tsv')
    try:  # if file exists but is empty
        mba_size = getsize(mg_mba_std)
    except:  # if file doesn't exist
        mba_size = -1
    if mba_size <= 0:
        logging.info('Calculating Coverage with jgi_summarize_bam_contig_depths\n')
        mba_cmd = ['jgi_summarize_bam_contig_depths', '--outputDepth', mg_mba_out]
        mba_cmd.extend(sorted_bam_list)
        with open(o_join(abr_path, mg_id + '.stderr.txt'), 'w') as stderr_file:
            run_mba = Popen(mba_cmd, stderr=stderr_file)
            run_mba.communicate()

        mg_mba_df = pd.read_csv(mg_mba_out, header=0, sep='\t')
        mg_mba_df.rename(columns={'contigName': 'subcontig_id'}, inplace=True)
        mg_mba_df.drop(columns=['contigLen', 'totalAvgDepth'], inplace=True)
        mg_mba_df.set_index('subcontig_id', inplace=True)
        scale = StandardScaler().fit(mg_mba_df.values)
        scaled_data = scale.transform(mg_mba_df.values)
        std_merge_df = pd.DataFrame(scaled_data, index=mg_mba_df.index).reset_index()
        std_merge_df.to_csv(mg_mba_std, header=True, sep='\t', index=False)

    return mg_mba_std


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


def runSAMSAM(abr_path, subcontig_path, mg_id, sam_list, nthreads):
    # run coverm on sorted bams
    mg_covm_out = o_join(abr_path, mg_id + '.ss_coverage.tsv')
    mg_covm_std = o_join(abr_path, mg_id + '.ss.scaled.tsv')
    try:  # if file exists but is empty
        covm_size = getsize(mg_covm_std)
    except:  # if file doesn't exist
        covm_size = -1
    if covm_size <= 0:
        for samfile in sam_list:
            logging.info('Calculating coverage on ' + samfile + ' with with SAMSUM\n')
            mg_asm = o_join(subcontig_path, mg_id + '.subcontigs.fasta')
            ref_seq_abunds = samsum_cmd.ref_sequence_abundances(aln_file=samfile, seq_file=mg_asm,
                                                                min_aln=10, p_cov=50,
                                                                map_qual=1, multireads=False
                                                                )
            refseq_merge_list = []
            for refseq_header in ref_seq_abunds.keys():
                refseq_obj = ref_seq_abunds[refseq_header]
                rso_name = refseq_obj.name
                rso_length = refseq_obj.length
                rso_reads_mapped = refseq_obj.reads_mapped
                rso_weight_total = refseq_obj.weight_total
                rso_fpkm = refseq_obj.fpkm
                rso_tpm = refseq_obj.tpm
                refseq_merge_list.append([rso_name, rso_length, rso_reads_mapped,
                                          rso_weight_total, rso_fpkm, rso_tpm
                                          ])
            mg_ss_df = pd.DataFrame(refseq_merge_list, columns=['subcontig_id', 'length',
                                                                'reads_mapped', 'weight_total',
                                                                'fpkm', 'tpm'
                                                                ])
            mg_ss_df.to_csv(mg_covm_out, sep='\t', index=False)
            mg_ss_df.drop(columns=['length', 'reads_mapped', 'weight_total', 'fpkm'], inplace=True)
            mg_ss_df.set_index('subcontig_id', inplace=True)
            scale = StandardScaler().fit(mg_ss_df.values)
            scaled_data = scale.transform(mg_ss_df.values)
            std_merge_df = pd.DataFrame(scaled_data, index=mg_ss_df.index).reset_index()
            std_merge_df.to_csv(mg_covm_std, header=True, sep='\t', index=False)

    return mg_covm_std


def runPySAM(abr_path, subcontig_path, mg_id, sorted_bam_list, nthreads):
    mg_asm = o_join(subcontig_path, mg_id + '.subcontigs.fasta')
    mg_seqs = s_utils.get_seqs(mg_asm)

    for bam in sorted_bam_list:
        samfile = pysam.AlignmentFile(bam, "rb")
        # cdef AlignedSegment read
        for read in samfile:
            1 + 1


def runBBtools(abr_path, subcontig_path, mg_id, raw_file_list, nthreads):
    pe1 = raw_file_list[0]
    mg_asm = o_join(subcontig_path, mg_id + '.subcontigs.fasta')
    mg_covm_std = o_join(abr_path, mg_id + '.coverage.scaled.tsv')

    if isfile(pe1) == True:
        pe_basename = basename(pe1)
        pe_id = pe_basename.split('.')[0]
        mg_sam_out = o_join(subcontig_path, pe_id + '.sam')
        if len(raw_file_list) == 2:
            logging.info('Raw reads in FWD and REV file...\n')
            pe2 = raw_file_list[1]
            in1 = ''.join(['in=', pe1])
            in2 = ''.join(['in2=', pe2])
            ref = ''.join(['ref=', mg_asm])
            ind_path = ''.join(['path=', subcontig_path])
            covstats = ''.join(['covstats=', o_join(subcontig_path, pe_id + '.covstats.txt')])
            covhist = ''.join(['covhist=', o_join(subcontig_path, pe_id + '.covhist.txt')])
            basecov = ''.join(['basecov=', o_join(subcontig_path, pe_id + '.basecov.txt')])
            bincov = ''.join(['bincov=', o_join(subcontig_path, pe_id + '.bincov.txt')])
            rpkm = ''.join(['rpkm=', o_join(subcontig_path, pe_id + '.rpkm.txt')])
            minid = ''.join(['minid=', str(0.76)])
            idfilter = ''.join(['idfilter=', str(0)])
            out = ''.join(['out=', mg_sam_out])
            threads = ''.join(['threads=', str(nthreads)])
            bb_cmd = ['bbmap.sh', in1, in2, ref, ind_path, covstats, covhist,
                      basecov, bincov, rpkm, minid, idfilter, out, threads,
                      'perfectmode=f']

        else:  # if the fastq is interleaved
            logging.info('Raw reads in interleaved file...\n')
            in1 = ''.join(['in=', pe1])
            ref = ''.join(['ref=', mg_asm])
            ind_path = ''.join(['path=', subcontig_path])
            covstats = ''.join(['covstats=', o_join(subcontig_path, pe_id + '.covstats.txt')])
            covhist = ''.join(['covhist=', o_join(subcontig_path, pe_id + '.covhist.txt')])
            basecov = ''.join(['basecov=', o_join(subcontig_path, pe_id + '.basecov.txt')])
            bincov = ''.join(['bincov=', o_join(subcontig_path, pe_id + '.bincov.txt')])
            rpkm = ''.join(['rpkm=', o_join(subcontig_path, pe_id + '.rpkm.txt')])
            minid = ''.join(['minid=', str(0.76)])
            idfilter = ''.join(['idfilter=', str(0)])
            out = ''.join(['out=', mg_sam_out])
            threads = ''.join(['threads=', str(nthreads)])
            bb_cmd = ['bbmap.sh', in1, ref, ind_path, covstats, covhist,
                      basecov, bincov, rpkm, minid, idfilter, out, threads,
                      'perfectmode=f']

        if isfile(covstats.split('=', 1)[1]) == False:
            logging.info('Running BBmap on %s\n' % pe_id)
            with open(mg_sam_out, 'w') as sam_file:
                with open(o_join(abr_path, pe_id + '.stderr.txt'), 'w') as stderr_file:
                    with open(o_join(abr_path, pe_id + '.stdout.txt'), 'w') as stdout_file:
                        run_mem = Popen(bb_cmd, stdout=stdout_file, stderr=stderr_file)
                        run_mem.communicate()
        else:
            print('SAM file already exists, skipping alignment...')
    else:
        print('Raw FASTQ file(s) are not where you said they were...')
        sys.exit()  # TODO: replace this quick-fix with a real exception

    print('Calculating Mean Adjusted Coverage and Variance...')
    chunk_iter = pd.read_csv(basecov.split('=', 1)[1], iterator=True, chunksize=100000,
                             header=0, sep='\t'
                             )
    basestats_list = []
    for i, chunk in tqdm(enumerate(chunk_iter)):
        if i == 0:
            tmp_chunk = chunk
        else:
            lead_chunk_df = tmp_chunk
            next_chunk_df = chunk
            lead_chunk_df.columns = ['subcontig_id', 'Pos', 'Coverage']
            next_chunk_df.columns = ['subcontig_id', 'Pos', 'Coverage']
            lead_list = lead_chunk_df['subcontig_id'].unique()
            lead_last = lead_chunk_df.iloc[-1, 0]
            next_first = next_chunk_df.iloc[0, 0]
            add_lead_df = next_chunk_df.query("subcontig_id in @lead_list")
            sub_next_df = next_chunk_df.query("subcontig_id != @lead_last")
            if add_lead_df.shape[0] != 0:
                concat_lead_df = pd.concat([lead_chunk_df, add_lead_df])
            else:
                concat_lead_df = lead_chunk_df
            tmp_chunk = sub_next_df
        if i != 0:
            chunk_stats = calcMeanVar(concat_lead_df)
            basestats_list.extend(chunk_stats)
    # Catch the last chunk on the way out :)
    lead_chunk_df = tmp_chunk
    next_chunk_df = chunk
    lead_chunk_df.columns = ['subcontig_id', 'Pos', 'Coverage']
    next_chunk_df.columns = ['subcontig_id', 'Pos', 'Coverage']
    concat_lead_df = pd.concat([lead_chunk_df, next_chunk_df])
    chunk_stats = calcMeanVar(concat_lead_df)
    basestats_list.extend(chunk_stats)
    # Build df and scale data
    basestats_df = pd.DataFrame(basestats_list, columns=['subcontig_id', 'mba_cov', 'mba_var'])
    # Filter out any subcontigs that had less than 50% coverage across it's length
    covstats_df = pd.read_csv(covstats.split('=', 1)[1], header=0, sep='\t')
    covstats_df.rename(columns={'#ID': 'subcontig_id'}, inplace=True)
    covstats_filter_df = covstats_df.copy()  # .query('Covered_percent > 50.0')  # TODO: might want to extract
    filter_list = list(covstats_filter_df['subcontig_id'].unique())
    basestats_filter_df = basestats_df.query('subcontig_id in @filter_list')
    # basestats_filter_df.set_index('subcontig_id', inplace=True)
    # scale = StandardScaler().fit(basestats_filter_df.values)
    # scaled_data = scale.transform(basestats_filter_df.values)
    # std_merge_df = pd.DataFrame(scaled_data, index=basestats_filter_df.index).reset_index()
    basestats_filter_df.to_csv(mg_covm_std, header=True, sep='\t', index=False)

    return mg_covm_std


def calcMeanVar(base_df):
    subbase_list = []
    for subcont in base_df['subcontig_id'].unique():
        subbase_df = base_df.query('subcontig_id == @subcont')
        minpos = subbase_df['Pos'].min() + 75
        maxpos = subbase_df['Pos'].max() - 75
        trimbase_df = subbase_df.query('Pos >= @minpos & Pos < @maxpos')
        submean = trimbase_df['Coverage'].mean()
        subvar = trimbase_df['Coverage'].var()
        subbase_list.append([subcont, submean, subvar])

    return subbase_list
