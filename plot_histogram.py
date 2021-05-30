import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances

import ot


def plot(filename, name, ax, label_x=0.95):
    print(filename)
    data = scipy.io.loadmat('data/{}'.format(filename))
    D = np.load('distance/{}.npy'.format(filename))

    n = D.shape[0]

    if 'X' in data:
        D[np.arange(n), np.arange(n)] += 1e18  # to avoid retrieving itself

    if 'X' in data:
        leftX = 'X'
        rightX = 'X'
        leftBOW = 'BOW_X'
        rightBOW = 'BOW_X'
    else:
        leftX = 'xte'
        rightX = 'xtr'
        leftBOW = 'BOW_xte'
        rightBOW = 'BOW_xtr'

    best = D.argmin(axis=1)
    Ds = []
    Ts = []
    for i in range(n):
        j = best[i]
        if data[leftX][0, i].shape[1] == 0 or data[rightX][0, j].shape[1] == 0:
            continue
        D = pairwise_distances(data[leftX][0, i].T, data[rightX][0, j].T)
        a = data[leftBOW][0, i][0].astype(float)
        a /= a.sum()
        b = data[rightBOW][0, j][0].astype(float)
        b /= b.sum()
        T = ot.emd(a, b, D)
        Ds.append(D.reshape(-1))
        Ts.append(T.reshape(-1))

    w = np.hstack(Ts)
    w = w / w.sum()

    ax.hist(np.hstack(Ds), weights=w, bins=20, color='#005aff')
    ax.set_xticks([0, 0.5, 1, 1.5])
    ax.tick_params(labelsize=16)
    ax.text(label_x, 0.95, name, size=24, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right')


if __name__ == '__main__':
    fig, ax = plt.subplots(2, 4, figsize=(28, 4))
    plot('bbcsport-emd_tr_te_split.mat', 'bbcsport', ax[0, 0])
    plot('twitter-emd_tr_te_split.mat', 'twitter', ax[0, 1])
    plot('recipe2-emd_tr_te_split.mat', 'recipe', ax[0, 2])
    plot('ohsumed-emd_tr_te_ix.mat', 'ohsumed', ax[0, 3], 0.5)
    plot('classic-emd_tr_te_split.mat', 'classic', ax[1, 0])
    plot('r8-emd_tr_te3.mat', 'reuters', ax[1, 1])
    plot('amazon-emd_tr_te_split.mat', 'amazon', ax[1, 2], 0.5)
    plot('20ng2_500-emd_tr_te.mat', '20news', ax[1, 3], 0.5)
    fig.tight_layout()
    fig.savefig('imgs/histogram.eps', bbox_inches='tight')
    fig.savefig('imgs/histogram.png', bbox_inches='tight')

    fig, ax = plt.subplots(2, 4, figsize=(28, 4))
    plot('bbcsport-emd_tr_te_split_5dim.mat', 'bbcsport_5dim', ax[0, 0])
    plot('twitter-emd_tr_te_split_5dim.mat', 'twitter_5dim', ax[0, 1])
    plot('recipe2-emd_tr_te_split_5dim.mat', 'recipe_5dim', ax[0, 2])
    plot('ohsumed-emd_tr_te_ix_5dim.mat', 'ohsumed_5dim', ax[0, 3])
    plot('classic-emd_tr_te_split_5dim.mat', 'classic_5dim', ax[1, 0])
    plot('r8-emd_tr_te3_5dim.mat', 'reuters_5dim', ax[1, 1])
    plot('amazon-emd_tr_te_split_5dim.mat', 'amazon_5dim', ax[1, 2])
    plot('20ng2_500-emd_tr_te_5dim.mat', '20news_5dim', ax[1, 3])
    fig.tight_layout()
    fig.savefig('imgs/histogram_5dim.eps', bbox_inches='tight')
    fig.savefig('imgs/histogram_5dim.png', bbox_inches='tight')
