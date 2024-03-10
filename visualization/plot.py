import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP

def plot_low_dimension(result, labels=None, savedir=None, alpha=1.0):
    
    os.makedirs(savedir, exist_ok=True)
    if labels is not None:
        label_cat = np.unique(labels)
    decomposition = {
        'tsne': TSNE(n_components=2, random_state=42),
        'umap': UMAP(n_components=2, random_state=42)
    }
    
    label_shape_map = {
          'train': 'o',  
          'valid': 'o',  # square
          'test': '^'  # triangle
      }

    for k in decomposition:
        plt.figure()
        flatten_mapping = decomposition[k].fit_transform(result)

        if labels is not None:
            for l in label_cat:
                marker = label_shape_map.get(l, 'o') 
                plt.scatter(flatten_mapping[labels == l, 0], flatten_mapping[labels == l, 1], label=l, alpha=alpha, marker=marker)
        else:
            plt.scatter(flatten_mapping[:, 0], flatten_mapping[:, 1], alpha=alpha)
        plt.legend()
        plt.title(k.upper())
        plt.savefig(f'./{savedir}/LM-{k}-opt_{alpha}.png')
