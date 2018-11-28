from dtreeviz.trees import *

know = pd.read_csv("data/knowledge.csv")
class_names = ['very_low', 'Low', 'Middle', 'High']
know['UNS'] = know['UNS'].map({n: i for i, n in enumerate(class_names)})

max_depth=3
x_train = know.PEG
y_train = know['UNS']
figsize = (6,2)
fig, ax = plt.subplots(1, 1, figsize=figsize)
ct = ctreeviz_univar(ax, x_train, y_train, max_depth=max_depth,
                     feature_name = 'PEG', class_names=class_names,
                     target_name='Knowledge',
                     nbins=40, gtype='strip',
                     show={'splits','title'})
plt.tight_layout()
plt.savefig(f"/tmp/knowledge-classtree-depth-{max_depth}.svg", bbox_inches=0, pad_inches=0)
plt.show()