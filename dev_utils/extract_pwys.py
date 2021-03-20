import os
import re
import sys
from collections import Counter

import pandas as pd


def extract_inference_pathways(in_path, additional_folder):
    pattern = "\(| |\)"
    pathways_dict = {}
    lst_output_folders = sorted([os.path.join(in_path, folder, additional_folder) for folder in os.listdir(in_path)
                                 if os.path.isdir(os.path.join(in_path, folder, additional_folder))])
    for idx, folder in enumerate(lst_output_folders):
        print(idx)
        print(folder)
        pgdb_id = folder.split('/')[-2]
        ptw_report_file = [os.path.join(folder, 'reports', f) for f in os.listdir(os.path.join(folder, 'reports'))
                           if str(f).startswith('pwy-inference-report')]
        if len(ptw_report_file) == 0:
            continue
        ptw_report_file = ptw_report_file[0]
        progress = (idx + 1) * 100.00 / len(lst_output_folders)
        print('\t{0:d})- Progress ({1:.2f}%): extracting information from: {2:s} folder...'
              .format(idx + 1, progress, os.path.basename(folder)))
        with open(ptw_report_file, 'r') as f_in:
            resume = False
            print('\t\t## Loading pathways report file from: {0:s}'.format(
                ptw_report_file))
            for aline in f_in:
                if not aline.startswith('List of pathways kept') and not resume:
                    continue
                if aline.startswith('List of pathways kept'):
                    resume = True
                    tmp = list()
                    continue
                if resume and not aline.startswith("\n"):
                    aline = list(filter(None, re.split(pattern, aline)))
                    aline = [str(i).strip() for i in aline if i != "\n"]
                    tmp.extend(aline)
                else:
                    resume = False
        lst_pathways = list(set([item for item in tmp if item != "\n"]))
        pathways_dict[pgdb_id] = lst_pathways
    return pathways_dict


pgdb_path = sys.argv[1]
output_file = sys.argv[2]
path_dict = extract_inference_pathways(in_path=pgdb_path, additional_folder="1.0")
path_df = pd.DataFrame({k: Counter(v) for k, v in path_dict.items()}).T.fillna(0).astype(int).reset_index()
path_df.rename(columns={path_df.columns[0]: "sampleID"}, inplace=True)
path_df.to_csv(output_file, sep='\t', index=False)
