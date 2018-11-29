from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from dtreeviz.trees import *

df_cars = pd.read_csv("data/cars.csv")
X = df_cars.drop('MPG', axis=1)
y = df_cars['MPG']

max_depth = 4
features = [2, 1]
X = X.values[:,features]
figsize = (6,5)
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111, projection='3d')

t = rtreeviz_bivar_3D(ax,
                      X, y,
                      max_depth=max_depth,
                      feature_names=['Vehicle Weight', 'Horse Power'],
                      target_name='MPG',
                      fontsize=14,
                      elev=20,
                      azim=25,
                      dist=8.2,
                      show={'splits','title'})
plt.savefig(f"/tmp/rtree-3D-depth-{max_depth}.svg", bbox_inches=0, pad_inches=0)

plt.show()