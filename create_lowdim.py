import scipy.io
import numpy as np
from sklearn.decomposition import PCA


def five(filename, dim):
    print(filename, dim)
    data = scipy.io.loadmat(filename)

    vs = []
    for v in data['X'][0]:
        if v.shape[1] > 0:
            vs.append(v.T)

    X = np.vstack(vs)

    pca = PCA(n_components=dim)
    X_pca = pca.fit_transform(X)

    cnt = 0
    for i, v in enumerate(data['X'][0]):
        if v.shape[1] > 0:
            l = data['X'][0, i].shape[1]
            data['X'][0, i] = X_pca[cnt:cnt+l].T
            cnt += l

    new_filename = filename[:-4] + '_{}dim.mat'.format(dim)

    scipy.io.savemat(new_filename, data)


def one(filename, dim):
    print(filename, dim)
    data = scipy.io.loadmat(filename)

    vs = []
    for v in data['xtr'][0]:
        if v.shape[1] > 0:
            vs.append(v.T)
    for v in data['xte'][0]:
        if v.shape[1] > 0:
            vs.append(v.T)

    X = np.vstack(vs)

    pca = PCA(n_components=dim)
    X_pca = pca.fit_transform(X)

    cnt = 0
    for i, v in enumerate(data['xtr'][0]):
        if v.shape[1] > 0:
            l = data['xtr'][0, i].shape[1]
            data['xtr'][0, i] = X_pca[cnt:cnt+l].T
            cnt += l
    for i, v in enumerate(data['xte'][0]):
        if v.shape[1] > 0:
            l = data['xte'][0, i].shape[1]
            data['xte'][0, i] = X_pca[cnt:cnt+l].T
            cnt += l

    new_filename = filename[:-4] + '_{}dim.mat'.format(dim)

    scipy.io.savemat(new_filename, data)


if __name__ == '__main__':
    five('data/bbcsport-emd_tr_te_split.mat', 5)
    five('data/twitter-emd_tr_te_split.mat', 5)
    five('data/recipe2-emd_tr_te_split.mat', 5)
    one('data/ohsumed-emd_tr_te_ix.mat', 5)
    five('data/classic-emd_tr_te_split.mat', 5)
    one('data/r8-emd_tr_te3.mat', 5)
    five('data/amazon-emd_tr_te_split.mat', 5)
    one('data/20ng2_500-emd_tr_te.mat', 5)
