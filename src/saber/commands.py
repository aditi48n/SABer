__author__ = 'Ryan J McLaughlin'

import logging

from pip._internal.operations import freeze

# import saber
import abundance_recruiter as abr
import classy as s_class
import clusterer as clst
import compile_recruits as com
import logger as s_log
import minhash_recruiter as mhr
import s_args
import tetranuc_recruiter as tra
import utilities as s_utils


def info(sys_args):
    """
    Function for writing version information about SABer and python dependencies.
    Other related info (citation, executable versions, etc.) should also be written through this sub-command.
    Create a SABerBase object for the `info` sub-command

    :param sys_args: List of arguments parsed from the command-line.
    :return: None
    """
    parser = s_args.SABerArgumentParser(description="Return package and executable information.")
    args = parser.parse_args(sys_args)
    s_log.prep_logging()
    info_s = s_class.SABerBase("info")

    logging.info("SABer version " + saber.version + ".\n")

    # Write the version of all python deps
    py_deps = {x.split('==')[0]: x.split('==')[1] for x in freeze.freeze()}

    logging.info("Python package dependency versions:\n\t" +
                 "\n\t".join([k + ": " + v for k, v in py_deps.items()]) + "\n")

    # Write the version of executable deps
    info_s.furnish_with_arguments(args)
    logging.info(s_utils.executable_dependency_versions(info_s.executables))  # TODO: needs updating for SABer exe

    if args.verbose:  # TODO: look at TS to determine what this is for.
        pass
        # logging.info(summary_str)

    return


def recruit(sys_args):
    """

    :param sys_args: List of arguments parsed from the command-line.
    :return: None
    """
    parser = s_args.SABerArgumentParser(description="Recruit environmental reads to reference contigs.")
    parser.add_recruit_args()
    args = parser.parse_args(sys_args)

    s_log.prep_logging("SABer_log.txt", args.verbose)
    recruit_s = s_class.SABerBase("recruit")
    recruit_s.trust_path = args.trust_path
    recruit_s.mg_file = args.mg_file
    recruit_s.mg_raw_file_list = args.mg_raw_file_list
    recruit_s.save_path = args.save_path
    recruit_s.max_contig_len = int(args.max_contig_len)
    recruit_s.overlap_len = int(args.overlap_len)
    recruit_s.nthreads = int(args.nthreads)
    recruit_s.force = args.force
    # Build save dir structure
    save_dirs_dict = s_utils.check_out_dirs(recruit_s.save_path)

    # Build subcontigs for MG
    mg_file = tuple([recruit_s.mg_file.rsplit('/', 1)[1].rsplit('.', 1)[0],
                     recruit_s.mg_file])  # TODO: needs to support multiple MetaGs
    mg_sub_file = s_utils.build_subcontigs('Metagenomes', [recruit_s.mg_file],
                                           save_dirs_dict['tmp'],
                                           recruit_s.max_contig_len,
                                           recruit_s.overlap_len
                                           )

    # Build minhash signatures if there are trusted contigs
    if recruit_s.trust_path:
        # Find the Trusted Contigs (TCs)
        tc_list = s_utils.get_SAGs(
            recruit_s.trust_path)  # TODO: needs to support a single multi-FASTA and multiple FASTAs
        trust_files = tuple([(x.rsplit('/', 1)[1].rsplit('.', 1)[0], x) for x in tc_list])
        # Run MinHash recruiting algorithm
        minhash_df_dict = mhr.run_minhash_recruiter(save_dirs_dict['tmp'],  # TODO: expose some params for users
                                                    save_dirs_dict['tmp'],
                                                    trust_files, mg_file,
                                                    recruit_s.nthreads, recruit_s.force
                                                    )
    else:
        minhash_df_dict = False

    # Build abundance tables
    abund_file = abr.runAbundRecruiter(save_dirs_dict['tmp'],
                                       save_dirs_dict['tmp'], mg_sub_file,
                                       recruit_s.mg_raw_file_list,
                                       recruit_s.nthreads, recruit_s.force
                                       )
    # Build tetra hz tables
    tetra_file = tra.run_tetra_recruiter(save_dirs_dict['tmp'],
                                         mg_sub_file
                                         )
    # Run HDBSCAN Cluster and Trusted Cluster Cleaning
    mg_id = mg_sub_file[0]
    clusters = clst.runClusterer(mg_id, save_dirs_dict['tmp'], abund_file, tetra_file,
                                 minhash_df_dict, recruit_s.nthreads
                                 )
    # Collect and join all recruits
    com.run_combine_recruits(save_dirs_dict['xPGs'], recruit_s.mg_file, clusters)

    return
