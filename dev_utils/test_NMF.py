import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

eurovision = pd.read_csv("~/Desktop/eurovision-2016.csv")
televote_Rank = eurovision.pivot(index='From country', columns='To country', values='Televote Rank')
# fill NAs by min per country
televote_Rank.fillna(televote_Rank.min(), inplace=True)
print(televote_Rank.head())
print(televote_Rank.shape)

# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components=2)
# Fit the model to televote_Rank
model.fit(televote_Rank)
# Transform the televote_Rank: nmf_features
nmf_features = model.transform(televote_Rank)
# Print the NMF features
nmf_feat_df = pd.DataFrame(nmf_features)
nmf_comp_df = pd.DataFrame(model.components_)
print(nmf_feat_df.head())
print(nmf_feat_df.shape)
print(nmf_comp_df.head())
print(nmf_comp_df.shape)

plt.figure(figsize=(20, 12))
countries = np.array(televote_Rank.index)
xs = nmf_features[:, 0]
print(xs[:5])
# Select the 1th feature: ys
ys = nmf_features[:, 1]
print(ys[:5])

# Scatter plot
plt.scatter(xs, ys, alpha=0.5)
# Annotate the points
for x, y, countries in zip(xs, ys, countries):
    plt.annotate(countries, (x, y), fontsize=10, alpha=0.5)
# plt.show()
