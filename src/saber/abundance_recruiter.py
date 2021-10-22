import logging
from os.path import isfile, basename, getsize
from os.path import join as o_join
from subprocess import Popen

import pandas as pd

import utilities as s_utils

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import StandardScaler
from samsum import commands as samsum_cmd
import sys


def runAbundRecruiter(subcontig_path, abr_path, mg_sub_file, mg_raw_file_list,
                      nthreads
                      ):
    logging.info('Starting Abundance Data Transformation\n')
    mg_id = mg_sub_file[0]
    if isfile(o_join(abr_path, mg_id + '.ss.scaled.tsv')):
        logging.info('Loading Abundance matrix for %s\n' % mg_id)
        mg_covm_out = o_join(abr_path, mg_id + '.ss.scaled.tsv')
    else:
        logging.info('Building %s abundance matrix\n' % mg_id)
        mg_sub_path = o_join(subcontig_path, mg_id + '.subcontigs.fasta')
        # Process raw metagenomes to calculate abundances
        mg_covm_out = procMetaGs(abr_path, mg_id, mg_raw_file_list,
                                 subcontig_path, nthreads
                                 )
    # Clean up the directory
    logging.info('Cleaning up intermediate files...\n')
    for s in ["*.sam", "*.bam", "*.stderr.txt", "*.stdout.txt", "*.ss_coverage.tsv"]:
        s_utils.runCleaner(abr_path, s)

    return mg_covm_out


def procMetaGs(abr_path, mg_id, mg_raw_file_list, subcontig_path, nthreads):
    # Process each raw metagenome
    with open(mg_raw_file_list, 'r') as raw_fa_in:
        raw_data = raw_fa_in.readlines()
    sam_list = []
    sorted_bam_list = []
    for line in raw_data:
        raw_file_list = line.strip('\n').split('\t')
        # Run BWA mem
        pe_id, mg_sam_out = runMiniMap2(abr_path, subcontig_path, mg_id, raw_file_list,
                                        nthreads
                                        )
        sam_list.append(mg_sam_out)
        # Build/sorted .bam files
        mg_sort_out = runSamTools(abr_path, pe_id, nthreads, mg_id, mg_sam_out)
        sorted_bam_list.append(mg_sort_out)
    logging.info('\n')
    # mg_covm_out = runCovM(abr_path, mg_id, nthreads, sorted_bam_list)
    mg_covm_out = runSAMSAM(abr_path, subcontig_path, mg_id, sam_list, nthreads, )

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
