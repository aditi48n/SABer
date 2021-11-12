__author__ = 'Ryan J McLaughlin'

import argparse
from argparse import RawTextHelpFormatter


class SABerArgumentParser(argparse.ArgumentParser):
    """
    A base argparse ArgumentParser for SABer with functions to furnish with common arguments.
    This standardizes the interface for a unified aesthetic across all sub-commands
    """

    def __init__(self, **kwargs):
        """
        Instantiate the argparse argument-parser and create three broad argument groups:
            reqs - for the required parameters
            optopt - for the optional parameters
            miscellany - for the miscellaneous parameters that are module agnostic,
            for example verbose, help, num_threads
        :param kwargs:
        """
        super(SABerArgumentParser, self).__init__(add_help=False,
                                                  formatter_class=RawTextHelpFormatter,
                                                  **kwargs
                                                  )
        self.reqs = self.add_argument_group("Required parameters")
        self.seqops = self.add_argument_group("Sequence operation arguments")
        self.optopt = self.add_argument_group("Optional options")
        self.miscellany = self.add_argument_group("Miscellaneous options")

        self.miscellany.add_argument("-v", "--verbose", action="store_true", default=False,
                                     help="Prints a more verbose runtime log")
        self.miscellany.add_argument("-h", "--help",
                                     action="help",
                                     help="Show this help message and exit")

    def parse_args(self, args=None, namespace=None):
        args = super(SABerArgumentParser, self).parse_args(args=args, namespace=namespace)

        return args

    def add_recruit_args(self):
        self.reqs.add_argument("-m", "--metag", required=True, dest="mg_file",
                               help="Path to a metagenome assembly [FASTA format only]."
                               )
        self.reqs.add_argument("-l", "--metaraw", required=True, dest="mg_raw_file_list",
                               help="Text file containing paths to raw FASTQ files for samples.\n"
                                    "One file per line, supports interleaved and separate PE reads.\n"
                                    "For separate PE files, both file paths on one line sep by [tab].\n"
                               )
        self.reqs.add_argument("-o", "--output-dir", required=True, dest="save_path",
                               help="Path to directory for all outputs."
                               )
        self.reqs.add_argument("-s", "--trusted-contigs", required=False, dest="trust_path",
                               default=False, help="Path to reference FASTA file or directory "
                                                   "containing only FASTA files."
                               )
        self.optopt.add_argument("--auto", action="store_true", dest="param_set",
                                 help="run SABer automatic optimization algorithm, this will\n"
                                      "likely provide better results than any others [Default]"
                                 )
        self.optopt.add_argument("--very_relaxed", action="store_true", dest="vr_params",
                                 help="parameter-set that maximizes recall at approximately strain-level\n"
                                      "[denovo_min_clust=50, denovo_min_samp=5\n"
                                      " anchor_min_clust=75, anchor_min_samp=10\n"
                                      " nu=0.7, gamma=10]"
                                 )
        self.optopt.add_argument("--relaxed", action="store_true", dest="r_params",
                                 help="parameter-set that maximizes recall at substrain-level\n"
                                      "[denovo_min_clust=50, denovo_min_samp=10\n"
                                      " anchor_min_clust=75, anchor_min_samp=10\n"
                                      " nu=0.7, gamma=10]"
                                 )
        self.optopt.add_argument("--strict", action="store_true", dest="s_params",
                                 help="parameter-set that maximizes precision at approximately strain-level\n"
                                      "[denovo_min_clust=75, denovo_min_samp=10\n"
                                      " anchor_min_clust=125, anchor_min_samp=10\n"
                                      " nu=0.3, gamma=0.1]"
                                 )
        self.optopt.add_argument("--very_strict", action="store_true", dest="vs_params",
                                 help="parameter-set that maximizes precision at substrain-level\n"
                                      "[denovo_min_clust=75, denovo_min_samp=10\n"
                                      " anchor_min_clust=125, anchor_min_samp=5\n"
                                      " nu=0.3, gamma=0.1]"
                                 )
        self.optopt.add_argument("--denovo_min_clust", required=False, dest="denovo_min_clust",
                                 help="minimum cluster size for De Novo HDBSCAN clustering."
                                 )
        self.optopt.add_argument("--anchor_min_clust", required=False, dest="anchor_min_clust",
                                 help="minimum cluster size for Anchored HDBSCAN clustering."
                                 )
        self.optopt.add_argument("--denovo_min_samp", required=False, dest="denovo_min_samp",
                                 help="minimum sample number for De Novo HDBSCAN clustering."
                                 )
        self.optopt.add_argument("--anchor_min_samp", required=False, dest="anchor_min_samp",
                                 help="minimum sample number for De Anchored HDBSCAN clustering."
                                 )
        self.optopt.add_argument("--nu", required=False, dest="nu",
                                 help="nu setting for Anchored OC-SVM clustering."
                                 )
        self.optopt.add_argument("--gamma", required=False, dest="gamma",
                                 help="gamma setting for Anchored OC-SVM clustering."
                                 )
        self.optopt.add_argument("--max_contig_len", required=False, default=10000,
                                 dest="max_contig_len",
                                 help="Max subcontig length in basepairs [10000]."
                                 )
        self.optopt.add_argument("--overlap_len", required=False, default=2000,
                                 dest="overlap_len",
                                 help="subcontig overlap in basepairs [2000]."
                                 )
        self.miscellany.add_argument("-t", "--num_threads", required=False, default=1,
                                     dest="nthreads",
                                     help="Number of threads [1]."
                                     )
        self.miscellany.add_argument("--force", required=False, default=False,
                                     action="store_true",
                                     help="Force SABer to run even if final recruits files exist [False]"
                                     )
        return
