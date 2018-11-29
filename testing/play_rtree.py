from dtreeviz.trees import *

df_cars = pd.read_csv("data/cars.csv")
X = df_cars.drop('MPG', axis=1)
y = df_cars['MPG']
X_train, y_train = X, y


max_depth = 3
fig = plt.figure()
ax = fig.gca()
t = rtreeviz_univar(ax,
                    X_train.WGT, y_train,
                    max_depth=max_depth,
                    feature_name='Vehicle Weight',
                    target_name='MPG',
                    fontsize=14,
                    show={'splits'})
plt.savefig(f"/tmp/cars-univar-{max_depth}.svg", bbox_inches=0, pad_inches=0)
plt.show()
