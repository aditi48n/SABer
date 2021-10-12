import itertools

import src.saber.abundance_recruiter as abr
import src.saber.clusterer as clst
import src.saber.minhash_recruiter as mhr
import src.saber.tetranuc_recruiter as tra

# This script is for running CV for HDBSCAN and OCSVM
# Expects that you have already pre-run the samples to create the
# coverage, tetra matrices, and feature embeddings (this was to save time)

nu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
gamma = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 'scale']
min_cluster_size = [15, 25, 50, 75, 100, 125, 150, 200]
min_samples = [5, 10, 15, 25, 35, 45, 55]
ocsvm_combo = list(itertools.product(*[nu, gamma]))
hdbscan_combo = list(itertools.product(*[min_cluster_size, min_samples]))

# Run MinHash recruiting algorithm
minhash_df_dict = mhr.run_minhash_recruiter(save_dirs_dict['tmp'],
                                            save_dirs_dict['tmp'],
                                            trust_files, mg_file,
                                            recruit_s.nthreads
                                            )

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
                             minhash_df_dict, 100, 25, 0.5, 'scale',
                             recruit_s.nthreads
                             )
