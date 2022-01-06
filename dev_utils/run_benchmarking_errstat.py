import os
import sys

sys.path.append('/home/ryan/dev/SABer')
sys.path.append('/home/ryan/dev/SABer/src')
sys.path.append('/home/ryan/dev/SABer/src/saber')
import vamb_errstat

# Input files directory
working_dir = sys.argv[1]
synthdata_dir = sys.argv[2]
mg_asm = sys.argv[3]
sample = sys.argv[4]
threads = int(sys.argv[5])

# SABer errstat
'''
mockpath = os.path.join(synthdata_dir, 'Final_SAGs_20k_test_subset/' + str(sample) + '/')
run_err_df = saber_errstat.runErrorAnalysis(working_dir, synthdata_dir, mg_asm, mockpath, threads)
run_err_df.to_csv(os.path.join(working_dir, 'Bin.errstat.tsv'), sep='\t', index=False)
print(run_err_df)
'''

# VAMB errstat
run_err_df = vamb_errstat.runErrorAnalysis(working_dir, synthdata_dir, mg_asm, threads)
run_err_df.to_csv(os.path.join(working_dir, 'Bin.errstat.tsv'), sep='\t', index=False)
print(run_err_df)
