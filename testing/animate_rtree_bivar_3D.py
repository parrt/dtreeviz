# This is broken

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from PIL import Image as PIL_Image
import glob
import cairosvg  # must pip install

from dtreeviz.trees import *

df_cars = pd.read_csv("data/cars.csv")
X = df_cars.drop('MPG', axis=1)
y = df_cars['MPG']

features = [2, 1]
X = X.values[:,features]

figsize = (6,5)
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111, projection='3d')

# start topdown view
t = rtreeviz_bivar_3D(ax,
                      X, y,
                      feature_names=['Vehicle Weight', 'Horse Power'],
                      target_name='MPG',
                      fontsize=14,
                      elev=90,
                      azim=0,
                      dist=7,
                      show={'splits'})

#plt.show()

n = 50
elev_range = np.arange(90, 10, -(90-10)/n)
azim_range = np.arange(0, 25, (22-0)/n)
i = 0

# pause on heatmap topview
for j in range(10):
    ax.elev = 90
    ax.azim = 0
    plt.savefig(f"/tmp/cars-frame-{i:02d}.png", bbox_inches=0, pad_inches=0, dpi=300)
    i += 1

# fly through
for elev, azim in zip(elev_range, azim_range):
    ax.elev = elev
    ax.azim = azim
    plt.savefig(f"/tmp/cars-frame-{i:02d}.png", bbox_inches=0, pad_inches=0, dpi=300)
    i += 1

# fly back
for elev, azim in reversed(list(zip(elev_range, azim_range))):
    ax.elev = elev
    ax.azim = azim
    plt.savefig(f"/tmp/cars-frame-{i:02d}.png", bbox_inches=0, pad_inches=0, dpi=300)
    i += 1

n_images = i

plt.close()

images = [PIL_Image.open(image) for image in [f'/tmp/cars-frame-{i:02d}.png' for i in range(n_images)]]
images[0].save('/tmp/cars-animation.gif',
               save_all=True,
               append_images=images[1:],
               duration=100,
               loop=0)
