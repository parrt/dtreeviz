from dtreeviz.trees import *

df_cars = pd.read_csv("data/cars.csv")
X = df_cars.drop('MPG', axis=1)
y = df_cars['MPG']

features=[2, 1]
X = X.values[:, features]
max_depth = 4
figsize = (6, 5)
fig, ax = plt.subplots(1, 1, figsize=figsize)
t = rtreeviz_bivar_heatmap(ax,
                           X, y,
                           max_depth=max_depth,
                           feature_names=['Vehicle Weight', 'Horse Power'],
                           fontsize=14)
plt.savefig(f"/tmp/rtree-heatmap-depth-{max_depth}.svg", bbox_inches=0, pad_inches=0)
plt.show()
