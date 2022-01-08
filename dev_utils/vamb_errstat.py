#!/usr/bin/env python

import glob
import logging
import multiprocessing
import sys
from functools import reduce
from os import makedirs, path
from os.path import join as joinpath

import pandas as pd
import pyfastx
from tqdm import tqdm

# specify that all columns should be shown
pd.set_option('max_columns', None)


def EArecruit(p):  # Error Analysis for all recruits per sag
    col_id, temp_id, temp_clust_df, temp_contig_df, temp_src2contig_list, \
    temp_src2strain_list, algorithm, src_id, strain_id, tot_bp_dict = p
    temp_clust_df[algorithm] = 1
    temp_contig_df[col_id] = temp_id
    df_list = [temp_contig_df, temp_clust_df]
    merge_recruits_df = reduce(lambda left, right: pd.merge(left, right,
                                                            on=[col_id, 'contig_id'],
                                                            how='left'),
                               df_list
                               )
    merge_recruits_df.fillna(-1, inplace=True)
    merge_recruits_df['exact_truth'] = [1 if x in temp_src2contig_list else -1
                                        for x in merge_recruits_df['contig_id']
                                        ]
    merge_recruits_df['strain_truth'] = [1 if x in temp_src2strain_list else -1
                                         for x in merge_recruits_df['contig_id']
                                         ]
    contig_id_list = list(merge_recruits_df['contig_id'])
    contig_bp_list = list(merge_recruits_df['bp_cnt'])
    exact_truth = list(merge_recruits_df['exact_truth'])
    strain_truth = list(merge_recruits_df['strain_truth'])
    src_total_bp = tot_bp_dict[src_id]
    algo_list = [algorithm]
    stats_lists = []
    for algo in algo_list:
        pred = list(merge_recruits_df[algo])
        rec_stats_list = recruit_stats([temp_id, algo, contig_id_list,
                                        contig_bp_list, exact_truth,
                                        strain_truth, pred, src_total_bp,
                                        src_id, strain_id
                                        ])
        stats_lists.extend(rec_stats_list)

    return stats_lists


def recruit_stats(p):
    sag_id, algo, contig_id_list, contig_bp_list, exact_truth, strain_truth, pred, tot_bp, \
    src_id, strain_id = p
    pred_df = pd.DataFrame(zip(contig_id_list, contig_bp_list, pred, ),
                           columns=['contig_id', 'contig_bp', 'pred']
                           )
    pred_df['sag_id'] = sag_id
    pred_df['algorithm'] = algo
    pred_df = pred_df[['sag_id', 'algorithm', 'contig_id', 'contig_bp', 'pred']]
    pred_df['truth'] = exact_truth
    pred_df['truth_strain'] = strain_truth
    # calculate for hybrid exact/strain-level matches
    TP = calc_tp(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    FP = calc_fp(pred_df['truth_strain'], pred_df['pred'], pred_df['contig_bp'])
    TN = calc_tn(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    FN = calc_fn(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    # compute total possible bp for each genome
    str_tot_bp_poss = TP + FN
    # Complete SRC genome is not always present in contigs, need to correct for that.
    working_bp = tot_bp - TP - FN
    FN = FN + working_bp
    str_list = calc_stats(sag_id, 'strain', algo, TP, FP, TN, FN,
                          pred_df['truth_strain'], pred_df['pred']
                          )
    # ALL Recruits
    # calculate for exact-level match
    TP = calc_tp(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    FP = calc_fp(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    TN = calc_tn(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    FN = calc_fn(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    # compute total possible bp for each genome
    exa_tot_bp_poss = TP + FN
    # Complete SRC genome is not always present in contigs, need to correct for that.
    working_bp = tot_bp - TP - FN
    FN = FN + working_bp
    x_list = calc_stats(sag_id, 'exact', algo, TP, FP, TN, FN,
                        pred_df['truth'], pred_df['pred']
                        )

    # Add total possible bp's for complete genome
    str_list.extend([str_tot_bp_poss, tot_bp])
    x_list.extend([exa_tot_bp_poss, tot_bp])

    cat_list = [str_list, x_list]

    return cat_list


def calc_tp(y_truth, y_pred, bp_cnt):
    tp_list = pd.Series([1 if ((x[0] == 1) & (x[1] == 1)) else 0 for x in zip(y_truth, y_pred)])
    tp_bp_list = pd.Series([x[0] * x[1] for x in zip(tp_list, bp_cnt)])
    TP = tp_bp_list.sum()

    return TP


def calc_fp(y_truth, y_pred, bp_cnt):
    fp_list = pd.Series([1 if ((x[0] == -1) & (x[1] == 1)) else 0 for x in zip(y_truth, y_pred)])
    fp_bp_list = pd.Series([x[0] * x[1] for x in zip(fp_list, bp_cnt)])
    FP = fp_bp_list.sum()

    return FP


def calc_tn(y_truth, y_pred, bp_cnt):
    tn_list = pd.Series([1 if ((x[0] == -1) & (x[1] == -1)) else 0 for x in zip(y_truth, y_pred)])
    tn_bp_list = pd.Series([x[0] * x[1] for x in zip(tn_list, bp_cnt)])
    TN = tn_bp_list.sum()

    return TN


def calc_fn(y_truth, y_pred, bp_cnt):
    fn_list = pd.Series([1 if ((x[0] == 1) & (x[1] == -1)) else 0 for x in zip(y_truth, y_pred)])
    fn_bp_list = pd.Series([x[0] * x[1] for x in zip(fn_list, bp_cnt)])
    FN = fn_bp_list.sum()

    return FN


def calc_stats(sag_id, level, algo, TP, FP, TN, FN, y_truth, y_pred):
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:
        sensitivity = 0
    else:
        sensitivity = TP / (TP + FN)
    N = TN + TP + FN + FP
    S = (TP + FN) / N
    P = (TP + FP) / N
    D = ((S * P) * (1 - S) * (1 - P)) ** (1 / 2)
    if D == 0:
        D = 1
    MCC = ((TP / N) - S * P) / D
    if precision + sensitivity == 0:
        F1 = 0
    else:
        F1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    # oc_precision, oc_recall, _ = precision_recall_curve(y_truth, y_pred)
    # AUC = auc(oc_recall, oc_precision)
    stat_list = [sag_id, level, algo, precision, sensitivity, MCC, F1,
                 N, S, P, TP, FP, TN, FN
                 ]

    return stat_list


def cnt_contig_bp(fasta_file):
    # counts basepairs/read contained in file
    # returns dictionary of {read_header:bp_count}

    fasta_records = get_seqs(fasta_file)
    fa_cnt_dict = {}
    for f_rec in fasta_records:
        fa_cnt_dict[f_rec.name] = len(f_rec.seq)
    return fa_cnt_dict


def cnt_total_bp(fasta_file):
    # counts total basepairs contained in file
    # returns fasta_file name and total counts for entire fasta file

    fasta_records = get_seqs(fasta_file)
    bp_sum = 0
    for f_rec in fasta_records:
        bp_sum += len(f_rec.seq)
    return fasta_file, bp_sum


def get_seqs(fasta_file):
    fasta = pyfastx.Fasta(fasta_file)
    return fasta


def cluster2taxonomy(p):
    clust, clust2src_df = p
    sub_clust_df = clust2src_df.query('best_label == @clust')
    exact_df = sub_clust_df.groupby(['CAMI_genomeID'])['bp_cnt'].sum().reset_index()
    strain_df = sub_clust_df.groupby(['strain'])['bp_cnt'].sum().reset_index()
    ex_label_df = exact_df[exact_df.bp_cnt == exact_df.bp_cnt.max()]['CAMI_genomeID']
    try:
        if not ex_label_df.empty:
            exact_label = exact_df[exact_df.bp_cnt == exact_df.bp_cnt.max()
                                   ]['CAMI_genomeID'].values[0]
            strain_label = strain_df[strain_df.bp_cnt == strain_df.bp_cnt.max()
                                     ]['strain'].values[0]
            return [clust, exact_label, strain_label]
    except:
        print(sub_clust_df.head())
        sys.exit()


def runErrorAnalysis(bin_path, synsrc_path, src_metag_file, nthreads):
    ##################################################################################################
    # INPUT files
    sag_tax_map = joinpath(synsrc_path, 'genome_taxa_info.tsv')
    mg_contig_map = joinpath(synsrc_path, 'gsa_mapping_pool.binning')
    src_contig_cnt = joinpath(synsrc_path, 'src_fasta.stats.tsv')
    src2id_map = joinpath(synsrc_path, 'genome_to_id.tsv')
    err_path = joinpath(bin_path, 'error_analysis')
    src2contig_file = joinpath(err_path, 'src2contig_map.tsv')
    denovo_out_file = glob.glob(joinpath(bin_path, 'clusters.tsv'))[0]
    denovo_errstat_file = joinpath(err_path, 'denovo.errstat.tsv')
    denovo_mean_file = joinpath(err_path, 'denovo.errstat.mean.tsv')

    ##################################################################################################
    # Make working dir
    if not path.exists(err_path):
        makedirs(err_path)

    # Map src contig stats to OTU id
    src_cnt_df = pd.read_csv(src_contig_cnt, sep='\t', header=0)
    src2id_df = pd.read_csv(src2id_map, sep='\t', header=None, names=['CAMI_genomeID', 'file'])
    src_cnt_df['src_id'] = [x.rsplit('/', 1)[1].rsplit('.', 1)[0] for x in src_cnt_df['file']]
    src2id_df['src_id'] = [x.rsplit('/', 1)[1].rsplit('.', 1)[0] for x in src2id_df['file']]
    src_stats_df = src2id_df.merge(src_cnt_df, on='src_id')

    # Map genome id and contig id to taxid for error analysis
    sag_taxmap_df = pd.read_csv(sag_tax_map, sep='\t', header=0)
    sag_taxmap_df['sp_taxid'] = [int(x) for x in sag_taxmap_df['@@TAXID']]
    sag_taxmap_df['sp_name'] = [x.split('|')[-2] for x in sag_taxmap_df['TAXPATHSN']]
    taxpath_list = [[str(x) for x in x.split('.')[0].split('|')]
                    for x in sag_taxmap_df['TAXPATH']
                    ]
    taxpath_df = pd.DataFrame(taxpath_list, columns=['domain', 'phylum', 'class', 'order',
                                                     'family', 'genus', 'species', 'strain'
                                                     ])
    taxpath_df['CAMI_genomeID'] = [x for x in sag_taxmap_df['_CAMI_genomeID']]
    # fix empty species id's
    taxpath_df['species'] = [x[1] if str(x[0]) == '' else x[0] for x in
                             zip(taxpath_df['species'], taxpath_df['genus'])
                             ]
    # Map MetaG contigs to their genomes
    mg_contig_map_df = pd.read_csv(mg_contig_map, sep='\t', header=0)
    # mg_contig_map_df['TAXID'] = [str(x) for x in mg_contig_map_df['TAXID']]

    # Merge contig map and taxpath DFs
    tax_mg_df = mg_contig_map_df.merge(taxpath_df, right_on='CAMI_genomeID', left_on='BINID',
                                       how='right'
                                       )
    tax_mg_df = tax_mg_df[['@@SEQUENCEID', 'CAMI_genomeID', 'domain', 'phylum', 'class', 'order',
                           'family', 'genus', 'species', 'strain'
                           ]]
    # count all bp's for Source genomes, Source MetaG, MockSAGs
    # count all bp's for each read in metaG
    src_metag_cnt_dict = cnt_contig_bp(src_metag_file)
    src_contig_list = list(src_metag_cnt_dict.keys())
    tax_mg_df = tax_mg_df.loc[tax_mg_df['@@SEQUENCEID'].isin(src_contig_list)]
    # Add to tax DF
    tax_mg_df['bp_cnt'] = [src_metag_cnt_dict[x] for x in tax_mg_df['@@SEQUENCEID']]
    # add src total bp count
    tax_mg_df = tax_mg_df.merge(src_stats_df[['CAMI_genomeID', 'sum_len']], on='CAMI_genomeID')
    tax_mg_df.to_csv(src2contig_file, sep='\t', index=False)

    # builds the sag to cami ID mapping file
    if 'CAMI_II' in synsrc_path:
        cami_genome2id_file = joinpath(synsrc_path, 'genome_to_id.tsv')
        cami_genome2id_df = pd.read_csv(cami_genome2id_file, sep='\t', header=None)
        cami_genome2id_df.columns = ['CAMI_genomeID', 'src_genome']
        cami_genome2id_df['src_genome'] = [x.rsplit('/', 1)[1].rsplit('.', 1)[0] for
                                           x in cami_genome2id_df['src_genome']
                                           ]
        cami_genome2id_dict = dict(zip(cami_genome2id_df['src_genome'],
                                       cami_genome2id_df['CAMI_genomeID']
                                       ))
    ###################################################################################################
    # De novo error analysis
    # setup mapping to CAMI ref genomes
    cluster_df = pd.read_csv(denovo_out_file, names=['best_label', 'contig_id'], sep='\t', header=None)
    cluster_trim_df = cluster_df.copy()  # .query('best_label != -1')
    src2contig_df = pd.read_csv(src2contig_file, header=0, sep='\t')
    src2contig_df = src2contig_df.rename(columns={'@@SEQUENCEID': 'contig_id'})
    src2contig_df['sample_id'] = [x.rsplit('C', 1)[0] for x in src2contig_df['contig_id']]
    contig_bp_df = src2contig_df[['contig_id', 'bp_cnt', 'sample_id']]
    clust2src_df = cluster_trim_df.merge(src2contig_df[['contig_id', 'CAMI_genomeID',
                                                        'strain', 'bp_cnt']],
                                         on='contig_id', how='left'
                                         )
    clust2src_df['sample_id'] = [x.rsplit('C', 1)[0] for x in clust2src_df['contig_id']]

    src_bp_dict = {x: y for x, y in zip(src2contig_df['CAMI_genomeID'], src2contig_df['sum_len'])}

    # Add taxonomy to each cluster
    clust_tax = []
    for clust in tqdm(clust2src_df['best_label'].unique()[:100]):
        samp_id = clust.rsplit('C', 1)[0]
        sub_clust2src_df = clust2src_df.query('sample_id == @samp_id')
        # arg_list.append([clust, sub_clust2src_df])
        clust_tax.append(cluster2taxonomy([clust, sub_clust2src_df]))
    clust_tax_df = pd.DataFrame(clust_tax, columns=['best_label', 'exact_label', 'strain_label'])
    clust2label_df = clust_tax_df.merge(cluster_trim_df, on='best_label', how='left')
    clust2contig_df = clust2label_df[['best_label', 'contig_id', 'exact_label', 'strain_label'
                                      ]].drop_duplicates()

    # setup multithreading pool
    print("De Novo error analysis started...")
    pool = multiprocessing.Pool(processes=nthreads)
    arg_list = []
    for clust in tqdm(clust2contig_df['best_label'].unique()[:100]):
        # subset recruit dataframes
        samp_id = clust.rsplit('C', 1)[0]
        sub_src2cont_df = src2contig_df.query('sample_id == @samp_id')
        sub_contig_bp_df = contig_bp_df.query('sample_id == @samp_id')
        sub_clust_df = clust2contig_df.query('best_label == @clust')
        dedup_clust_df = sub_clust_df[['best_label', 'contig_id']].drop_duplicates()
        # Map Sources/SAGs to Strain IDs
        src_id = sub_clust_df['exact_label'].values[0]
        strain_id = sub_clust_df['strain_label'].values[0]
        src_sub_df = sub_src2cont_df.query('CAMI_genomeID == @src_id')
        strain_sub_df = sub_src2cont_df.query('strain == @strain_id')
        src2contig_list = list(set(src_sub_df['contig_id'].values))
        src2strain_list = list(set(strain_sub_df['contig_id'].values))
        arg_list.append(['best_label', clust, dedup_clust_df, sub_contig_bp_df, src2contig_list,
                         src2strain_list, 'denovo', src_id, strain_id, src_bp_dict
                         ])

    results = pool.imap_unordered(EArecruit, arg_list)
    score_list = []
    for i, output in tqdm(enumerate(results, 1)):
        score_list.extend(output)
    logging.info('\n')
    pool.close()
    pool.join()

    score_df = pd.DataFrame(score_list, columns=['best_label', 'level', 'algorithm',
                                                 'precision', 'sensitivity', 'MCC', 'F1',
                                                 'N', 'S', 'P', 'TP', 'FP', 'TN', 'FN',
                                                 'possible_bp', 'total_bp'
                                                 ])

    sort_score_df = score_df.sort_values(['best_label', 'level', 'precision', 'sensitivity'],
                                         ascending=[False, False, True, True]
                                         )
    score_tax_df = sort_score_df.merge(clust_tax_df, on='best_label', how='left')
    score_tax_df['size_bp'] = score_tax_df['TP'] + score_tax_df['FP']
    score_tax_df['>20Kb'] = 'No'
    score_tax_df.loc[score_tax_df['size_bp'] >= 20000, '>20Kb'] = 'Yes'
    score_tax_df['NC_bins'] = 'No'
    score_tax_df.loc[(score_tax_df['precision'] >= 0.95) &
                     (score_tax_df['sensitivity'] >= 0.9), 'NC_bins'] = 'Yes'
    score_tax_df['MQ_bins'] = 'No'
    score_tax_df.loc[(score_tax_df['precision'] >= 0.9) &
                     (score_tax_df['sensitivity'] >= 0.5), 'MQ_bins'] = 'Yes'
    # possible bp's based on asm vs ref genome
    poss_bp_df = score_tax_df[['exact_label', 'possible_bp', 'total_bp']].copy().drop_duplicates()
    poss_bp_df['asm_per_bp'] = [x / y for x, y in
                                zip(poss_bp_df['possible_bp'],
                                    poss_bp_df['total_bp'])
                                ]
    poss_bp_df['yes_NC'] = [1 if x >= 0.9 else 0 for x in poss_bp_df['asm_per_bp']]
    poss_bp_df['yes_MQ'] = [1 if x >= 0.5 else 0 for x in poss_bp_df['asm_per_bp']]
    poss_sum_df = poss_bp_df.groupby().sum().reset_index()
    print(poss_bp_df.head())
    print(poss_sum_df)
    sys.exit()
    # add possible bp to score df
    score_tax_df = score_tax_df.merge(poss_bp_df[['exact_label', 'yes_NC', 'yes_MQ']],
                                      on='exact_lebel', how='left'
                                      )
    stat_mean_df = score_tax_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                         'MQ_bins'])[['precision', 'sensitivity', 'MCC',
                                                      'F1']].mean().reset_index()
    cnt_bins_df = score_tax_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                        'MQ_bins']).size().reset_index()
    cnt_bins_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                           'bin_cnt'
                           ]
    cnt_exact_df = score_tax_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                         'MQ_bins'])[['exact_label']
    ].nunique().reset_index()
    cnt_exact_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                            'genome_cnt'
                            ]
    cnt_genos_df = score_tax_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                         'MQ_bins'])[['strain_label']
    ].nunique().reset_index()
    cnt_genos_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                            'strain_cnt'
                            ]
    dfs = [stat_mean_df, cnt_bins_df, cnt_exact_df, cnt_genos_df]
    stat_df = reduce(lambda left, right: pd.merge(left, right, on=['level',
                                                                   'algorithm', '>20Kb',
                                                                   'NC_bins', 'MQ_bins'
                                                                   ]), dfs
                     )
    score_tax_df.to_csv(denovo_errstat_file, index=False, sep='\t')
    stat_df.to_csv(denovo_mean_file, index=False, sep='\t')

    ###########################################################################
    # Compile all results tables from analysis
    completed_files = glob.glob(joinpath(err_path, '*.errstat.tsv'))
    cat_list = []
    for o_file in completed_files:
        err_df = pd.read_csv(o_file, sep='\t', header=0)
        for algo in err_df['algorithm'].unique():
            for level in err_df['level'].unique():
                sub_err_df = err_df.query('algorithm == @algo & level == @level')
                l_20 = '>20Kb'
                mq_df = sub_err_df.query("NC_bins == 'Yes' | MQ_bins == 'Yes' | @l_20 == 'Yes'")
                nc_df = sub_err_df.query("NC_bins == 'Yes' | @l_20 == 'Yes'")
                mq_avg_mcc = mq_df['MCC'].mean()
                nc_avg_mcc = nc_df['MCC'].mean()
                mq_avg_p = mq_df['precision'].mean()
                nc_avg_p = nc_df['precision'].mean()
                mq_avg_r = mq_df['sensitivity'].mean()
                nc_avg_r = nc_df['sensitivity'].mean()
                mq_cnt = mq_df['MQ_bins'].count()
                nc_cnt = nc_df['NC_bins'].count()
                err_list = [algo, level, mq_avg_p, mq_avg_r, mq_avg_mcc,
                            mq_cnt, nc_avg_p, nc_avg_r, nc_avg_mcc, nc_cnt
                            ]
                cat_list.append(err_list)

    cat_cols = ['algo', 'level', 'mq_avg_p', 'mq_avg_r', 'mq_avg_mcc',
                'mq_cnt', 'nc_avg_p', 'nc_avg_r', 'nc_avg_mcc', 'nc_cnt'
                ]
    # add the total possible NC and MQ bins
    cat_df = pd.DataFrame(cat_list, columns=cat_cols)
    poss_bp_df.columns = ['level', 'algo', 'NC_possible', 'MQ_possible']
    cat_bp_df = cat_df.merge(poss_bp_df, on=['level', 'algo'],
                             how='left'
                             )
    return cat_bp_df
