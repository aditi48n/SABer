import sys
import pandas as pd
from os import makedirs, path, listdir
from os.path import join as joinpath
import shutil


input_dir = sys.argv[1]
output_dir = sys.argv[2]
sag2cami_file = sys.argv[3]

# Make output dir
if not path.exists(output_dir):
    makedirs(output_dir)

file_list = [x for x in listdir(input_dir) if x.rsplit('.', 1)[-1] == 'fasta']
sag_list = [x.rsplit('.', 1)[0] for x in file_list]
sag_file_dict = dict(zip(sag_list, file_list))
sag2cami_df = pd.read_csv(sag2cami_file, header=0, sep='\t')

split_dat_dict = {0: [], 1: [], 2: [], 3: [], 4: [],
                  5: [], 6: [], 7: [], 8: [], 9: []
                  }
for cami in sag2cami_df['CAMI_genomeID'].unique():
    sub_cami_df = sag2cami_df.query('CAMI_genomeID == @cami')
    sag_list = set(sub_cami_df['sag_id'].unique())
    for i, sag_id in enumerate(sag_list):
        split_dat_dict[i].append(sag_id)
for k, v in split_dat_dict.items():
    print(k, len(v))
    sub_dir = joinpath(output_dir, str(k))
    # Make subdirs dir
    if not path.exists(sub_dir):
        makedirs(sub_dir)
    for f in v:
        fa_file = sag_file_dict[f]
        src = joinpath(input_dir, fa_file)
        dst = joinpath(sub_dir, fa_file)
        shutil.copy(src, dst)
