import numpy as np
import scipy.sparse
import scipy.io
from sklearn.metrics import pairwise_distances
from util import load, load_one


def check_five(filename, name):
    print(filename)
    data, X, y = load('data/{}'.format(filename))
    n = X.shape[0]
    D = pairwise_distances(X)
    D[np.arange(n), np.arange(n)] = 1
    dup = (D == 0)
    print('duplicate pair:', dup.sum() // 2)
    print('duplicate sample:', (dup.sum(1) > 0).sum())
    print(np.where(D == 0))

    mask = np.ones(n, dtype=int)
    np.random.seed(0)
    for i in np.random.permutation(n):
        if (dup[i] * mask).sum() > 0:
            mask[i] = 0
    ind = np.arange(n)[mask == 1]
    ma = {x: i for i, x in enumerate(ind)}

    data['X'] = data['X'][:, ind]
    data['BOW_X'] = data['BOW_X'][:, ind]
    data['Y'] = data['Y'][:, ind]

    tr = []
    te = []
    for i in range(5):
        train = data['TR'][i] - 1
        test = data['TE'][i] - 1
        train = np.array(sorted([ma[x] for x in train if x in ma]))
        test = np.array(sorted([ma[x] for x in test if x in ma]))
        tr.append(train + 1)
        te.append(test + 1)
    data['TR'] = np.array(tr, dtype=object)
    data['TE'] = np.array(te, dtype=object)

    scipy.io.savemat('data/{}_clean.mat'.format(name), data)

    D = np.load('distance/{}.npy'.format(filename))
    D_tfidf = np.load('distance/{}-tfidf.npy'.format(filename))

    D = D[ind][:, ind]
    D_tfidf = D_tfidf[ind][:, ind]

    np.save('distance/{}_clean.mat.npy'.format(name), D)
    np.save('distance/{}_clean.mat-tfidf.npy'.format(name), D_tfidf)


def check_one(filename, name):
    print(filename)
    X_train, y_train, X_test, y_test = load_one('data/{}'.format(filename))
    data = scipy.io.loadmat('data/{}'.format(filename))
    X = scipy.sparse.vstack([X_train, X_test])
    n = X.shape[0]
    D = pairwise_distances(X)
    D[np.arange(n), np.arange(n)] = 1
    dup = (D == 0)
    print('duplicate pair:', dup.sum() // 2)
    print('duplicate sample:', (dup.sum(1) > 0).sum())
    print(np.where(D == 0))

    mask = np.ones(n, dtype=int)
    np.random.seed(0)
    for i in np.random.permutation(n):
        if (dup[i] * mask).sum() > 0:
            mask[i] = 0
    ind = np.arange(n)[mask == 1]
    ind_train = ind[ind < X_train.shape[0]]
    ind_test = ind[ind >= X_train.shape[0]] - X_train.shape[0]
    data['xtr'] = data['xtr'][:, ind_train]
    data['xte'] = data['xte'][:, ind_test]
    data['BOW_xtr'] = data['BOW_xtr'][:, ind_train]
    data['BOW_xte'] = data['BOW_xte'][:, ind_test]
    data['ytr'] = data['ytr'][:, ind_train]
    data['yte'] = data['yte'][:, ind_test]

    scipy.io.savemat('data/{}_clean.mat'.format(name), data)

    D = np.load('distance/{}.npy'.format(filename))
    D_train = np.load('distance/{}-train.npy'.format(filename))
    D_tfidf = np.load('distance/{}-tfidf.npy'.format(filename))
    D_train_tfidf = np.load('distance/{}-train-tfidf.npy'.format(filename))

    D = D[ind_test][:, ind_train]
    D_train = D_train[ind_train][:, ind_train]
    D_tfidf = D_tfidf[ind_test][:, ind_train]
    D_train_tfidf[ind_train][:, ind_train]

    np.save('distance/{}_clean.mat.npy'.format(name), D)
    np.save('distance/{}_clean.mat-train.npy'.format(name), D_train)
    np.save('distance/{}_clean.mat-tfidf.npy'.format(name), D_tfidf)
    np.save('distance/{}_clean.mat-train-tfidf.npy'.format(name), D_train_tfidf)


if __name__ == '__main__':
    check_five('bbcsport-emd_tr_te_split.mat', 'bbcsport')
    check_five('twitter-emd_tr_te_split.mat', 'twitter')
    check_five('recipe2-emd_tr_te_split.mat', 'recipe')
    check_one('ohsumed-emd_tr_te_ix.mat', 'ohsumed')
    check_five('classic-emd_tr_te_split.mat', 'classic')
    check_one('r8-emd_tr_te3.mat',  'reuter')
    check_five('amazon-emd_tr_te_split.mat', 'amazon')
    check_one('20ng2_500-emd_tr_te.mat', '20news')
