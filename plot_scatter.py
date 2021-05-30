import numpy as np
import matplotlib.pyplot as plt

from util import load, load_one

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize


def correlation(filename, name, fold, ax):
    print(filename)
    if fold == 'five':
        data, X, y = load('data/{}'.format(filename))
        X = normalize(X, axis=1, norm='l1')
        D_bow = pairwise_distances(X, metric='manhattan')
    elif fold == 'one':
        X_train, y_train, X_test, y_test = load_one('data/{}'.format(filename))
        X_train = normalize(X_train, axis=1, norm='l1')
        X_test = normalize(X_test, axis=1, norm='l1')
        D_bow = pairwise_distances(X_test, X_train, metric='manhattan')

    D_wmd = np.load('distance/{}.npy'.format(filename))

    D_wmd_ind = D_wmd.reshape(-1) >= 0

    corr = np.corrcoef(D_bow.reshape(-1)[D_wmd_ind], D_wmd.reshape(-1)[D_wmd_ind])[0, 1]

    ax.scatter(D_bow.reshape(-1)[D_wmd_ind], D_wmd.reshape(-1)[D_wmd_ind], c='#005aff', s=1, rasterized=True)
    ax.tick_params(labelsize=16)
    ax.set_xlabel('BOW (L1/L1)', size=20)
    ax.set_ylabel('WMD', size=20)
    ax.text(0.05, 0.95, '{}\n$\\rho = {:.3f}$'.format(name, corr), size=24, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left')


if __name__ == '__main__':
    fig, ax = plt.subplots(2, 4, figsize=(24, 12))
    correlation('bbcsport-emd_tr_te_split.mat', 'bbcsport', 'five', ax[0, 0])
    correlation('twitter-emd_tr_te_split.mat', 'twitter', 'five', ax[0, 1])
    correlation('recipe2-emd_tr_te_split.mat', 'recipe', 'five', ax[0, 2])
    correlation('ohsumed-emd_tr_te_ix.mat', 'ohsumed', 'one', ax[0, 3])
    correlation('classic-emd_tr_te_split.mat', 'classic', 'five', ax[1, 0])
    correlation('r8-emd_tr_te3.mat', 'reuters', 'one', ax[1, 1])
    correlation('amazon-emd_tr_te_split.mat', 'amazon', 'five', ax[1, 2])
    correlation('20ng2_500-emd_tr_te.mat', '20news', 'one', ax[1, 3])
    fig.tight_layout()
    fig.savefig('imgs/scatter.eps', bbox_inches='tight', dpi=96)
    fig.savefig('imgs/scatter.png', bbox_inches='tight', dpi=96)

    fig, ax = plt.subplots(2, 4, figsize=(24, 12))
    correlation('bbcsport-emd_tr_te_split_5dim.mat', 'bbcsport_5dim', 'five', ax[0, 0])
    correlation('twitter-emd_tr_te_split_5dim.mat', 'twitter_5dim', 'five', ax[0, 1])
    correlation('recipe2-emd_tr_te_split_5dim.mat', 'recipe_5dim', 'five', ax[0, 2])
    correlation('ohsumed-emd_tr_te_ix_5dim.mat', 'ohsumed_5dim', 'one', ax[0, 3])
    correlation('classic-emd_tr_te_split_5dim.mat', 'classic_5dim', 'five', ax[1, 0])
    correlation('r8-emd_tr_te3_5dim.mat', 'reuters_5dim', 'one', ax[1, 1])
    correlation('amazon-emd_tr_te_split_5dim.mat', 'amazon_5dim', 'five', ax[1, 2])
    correlation('20ng2_500-emd_tr_te_5dim.mat', '20news_5dim', 'one', ax[1, 3])
    fig.tight_layout()
    fig.savefig('imgs/scatter_5dim.eps', bbox_inches='tight', dpi=96)
    fig.savefig('imgs/scatter_5dim.png', bbox_inches='tight', dpi=96)
