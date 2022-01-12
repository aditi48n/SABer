import os
import sys

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# plot aestetics
sns.set_context("paper")

saber_single_file = sys.argv[1]
saber_multi_file = sys.argv[2]
unitem_single_file = sys.argv[3]
unitem_multi_file = sys.argv[4]
# vamb_multi_file = sys.argv[5] # when it's ready

# working directory
workdir = os.path.dirname(saber_single_file)

# column renaming/mapping dictionaries
type2label = {'CAMI_II_Airways': 'Air',
              'CAMI_II_Gastrointestinal': 'GI',
              'CAMI_II_Oral': 'Oral',
              'CAMI_II_Skin': 'Skin',
              'CAMI_II_Urogenital': 'Urog'
              }
algo2rank = {'denovo': 0, 'hdbscan': 1,
             'ocsvm': 2, 'intersect': 3
             }
type2rank = {'CAMI_II_Airways': 0,
             'CAMI_II_Gastrointestinal': 1,
             'CAMI_II_Oral': 2,
             'CAMI_II_Skin': 3,
             'CAMI_II_Urogenital': 4
             }

# Munge SABer single/mulit
# Near Complete
ss_df = pd.read_csv(saber_single_file, header=0, sep='\t')
ss_df['label'] = [type2label[x] for x in ss_df['sample_type']]
ss_df['algo_rank'] = [algo2rank[x] for x in ss_df['algo']]
ss_df['type_rank'] = [type2rank[x] for x in ss_df['sample_type']]
ss_abs_str_df = ss_df.query("level == 'strain_absolute'")
ss_abs_str_df.sort_values(by=['type_rank', 'algo_rank'], inplace=True)

# Medium Quality
sm_df = pd.read_csv(saber_multi_file, header=0, sep='\t')
sm_df['label'] = [type2label[x] for x in sm_df['sample_type']]
sm_df['algo_rank'] = [algo2rank[x] for x in sm_df['algo']]
sm_df['type_rank'] = [type2rank[x] for x in sm_df['sample_type']]
sm_abs_str_df = sm_df.query("level == 'strain_absolute'")
sm_abs_str_df.sort_values(by=['type_rank', 'algo_rank'], inplace=True)

# Boxplots for mode and param set
ss_box = sns.catplot(x="mode", y="ext_nc_uniq", hue="param_set",
                     kind="bar", data=ss_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.single.mode_param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()
sm_box = sns.catplot(x="mode", y="ext_nc_uniq", hue="param_set",
                     kind="bar", data=sm_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
sm_box.savefig(os.path.join(workdir, 'SABer.multi.mode_param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

sys.exit()

# Build boxplots
ss_box = sns.catplot(x="label", y="ext_nc_uniq", hue="algo",
                     kind="box", data=ss_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.single.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()
sm_box = sns.catplot(x="label", y="ext_nc_uniq", hue="algo",
                     kind="box", data=sm_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
sm_box.savefig(os.path.join(workdir, 'SABer.multi.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

print(ss_df.head())
print(sm_df.head())
