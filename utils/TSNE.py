import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def get_tsne(train_data, n_components=2):
    data_1d = train_data.reshape(train_data.shape[0], -1)
    tsne = TSNE(n_components=n_components, verbose=0, perplexity=50, n_iter=500)

    tsne_results = tsne.fit_transform(data_1d)
    return tsne, tsne_results

def scatter_tsne(tsne_result, label, axis=None):
    assert tsne_result.shape[1] == 2
    scatter = sns.scatterplot(
        x=tsne_result[:,0], y=tsne_result[:,1],
        hue=label.astype(np.int8),
        palette=sns.color_palette("hls", 10),
        legend="full",
        alpha=0.5,
        ax=axis,
    )
    return scatter