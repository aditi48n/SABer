import sys

import pandas as pd

saber_single_file = sys.argv[1]
saber_multi_file = sys.argv[2]
unitem_single_file = sys.argv[3]
unitem_multi_file = sys.argv[4]
# vamb_multi_file = sys.argv[5] # when it's ready

# Munge SABer single/mulit
ss_df = pd.read_csv(saber_single_file, header=0, sep='\t')
sm_df = pd.read_csv(saber_multi_file, header=0, sep='\t')

print(ss_df.head())
print(sm_df.head())
