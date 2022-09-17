import logging
import warnings
from os.path import isfile
from os.path import join as o_join

import saber.utilities as s_utils

warnings.simplefilter(action='ignore', category=FutureWarning)


def run_tetra_recruiter(tra_path, mg_sub_file):
    logging.info('Starting Tetranucleotide Data Transformation\n')
    mg_id = mg_sub_file[0]
    if isfile(o_join(tra_path, mg_id + '.tetras.tsv')):
        logging.info('Loading tetramer Hz matrix for %s\n' % mg_id)
        mg_tetra_file = o_join(tra_path, mg_id + '.tetras.tsv')
    else:
        logging.info('Calculating tetramer Hz matrix for %s\n' % mg_id)
        mg_subcontigs = s_utils.get_seqs(mg_sub_file[1])
        mg_tetra_df = s_utils.tetra_cnt(mg_subcontigs)
        mg_tetra_df.to_csv(o_join(tra_path, mg_id + '.tetras.tsv'),
                           sep='\t'
                           )
        mg_tetra_file = o_join(tra_path, mg_id + '.tetras.tsv')

    return mg_tetra_file
