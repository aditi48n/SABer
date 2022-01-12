import os
import sys

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

saber_single_file = sys.argv[1]
saber_multi_file = sys.argv[2]
unitem_single_file = sys.argv[3]
unitem_multi_file = sys.argv[4]
# vamb_multi_file = sys.argv[5] # when it's ready

# working directory
workdir = os.path.dirname(saber_single_file)

# Munge SABer single/mulit
ss_df = pd.read_csv(saber_single_file, header=0, sep='\t')
ss_abs_str_df = ss_df.query("level == strain_absolute")
sm_df = pd.read_csv(saber_multi_file, header=0, sep='\t')
sm_abs_str_df = sm_df.query("level == strain_absolute")

# Build boxplots
ss_box = sns.catplot(x="sample_type", y="ext_nc_uniq", hue="algo", kind="box", data=ss_abs_str_df)
sm_box = sns.catplot(x="sample_type", y="ext_nc_uniq", hue="algo", kind="box", data=sm_abs_str_df)
ss_box.savefig(os.path.join(workdir, 'SABer.single.boxplot.png'), dpi=300)
plt.clf()
plt.close()

print(ss_df.head())
print(sm_df.head())
