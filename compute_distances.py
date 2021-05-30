import os
import argparse

import scipy.io
import numpy as np
from sklearn.metrics import pairwise_distances
import pickle

import ot

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('split', type=int)
parser.add_argument('part', type=int)
parser.add_argument('--train', action='store_true')
parser.add_argument('--tfidf', action='store_true')
args = parser.parse_args()

mainname = args.filename.split('/')[-1]
dtrain = 'train' if args.train else 'none'
dtfidf = 'tfidf' if args.tfidf else 'bow'

data = scipy.io.loadmat(args.filename)

if 'X' in data:
    X = np.vstack([x.T for x in data['X'][0]])
    _, inverse = np.unique(X, axis=0, return_inverse=True)
    freq = np.bincount(inverse)
    N = len(data['X'][0])
else:
    X = np.vstack([x.T for x in data['xtr'][0]] + [x.T for x in data['xte'][0] if len(x.T) > 0])
    _, inverse = np.unique(X, axis=0, return_inverse=True)
    freq = np.bincount(inverse)
    N = len(data['xtr'][0]) + len(data['xte'][0])

if 'X' in data:
    leftX = 'X'
    rightX = 'X'
    leftBOW = 'BOW_X'
    rightBOW = 'BOW_X'
    leftind = np.cumsum([0] + [x.shape[1] for x in data['X'][0]])
    rightind = np.cumsum([0] + [x.shape[1] for x in data['X'][0]])
elif dtrain == 'train':
    leftX = 'xtr'
    rightX = 'xtr'
    leftBOW = 'BOW_xtr'
    rightBOW = 'BOW_xtr'
    leftind = np.cumsum([0] + [x.shape[1] for x in data['xtr'][0]])
    rightind = np.cumsum([0] + [x.shape[1] for x in data['xtr'][0]])
else:
    leftX = 'xte'
    rightX = 'xtr'
    leftBOW = 'BOW_xte'
    rightBOW = 'BOW_xtr'
    rightind = np.cumsum([0] + [x.shape[1] for x in data['xtr'][0]])
    leftind = np.cumsum([rightind[-1]] + [x.shape[1] for x in data['xte'][0]])


n = len(data[leftX][0])
m = len(data[rightX][0])

pair_list = [(i, j) for i in range(n) for j in range(m)]

ids = [0]
for i in range(args.split):
    nex = ids[-1] + len(pair_list) // args.split
    if i < len(pair_list) % args.split:
        nex += 1
    ids.append(nex)

assert(ids[-1] == len(pair_list))

start = ids[args.part]
end = ids[args.part+1]

vals = []
for k, (i, j) in enumerate(pair_list[start:end]):
    if data[leftX][0, i].shape[1] == 0 or data[rightX][0, j].shape[1] == 0:
        vals.append((i, j, -1))
        continue
    D = pairwise_distances(data[leftX][0, i].T, data[rightX][0, j].T)
    a = data[leftBOW][0, i][0].astype(float)
    b = data[rightBOW][0, j][0].astype(float)
    if args.tfidf:
        a = a * np.log(N / freq[inverse[leftind[i]:leftind[i+1]]])
        b = b * np.log(N / freq[inverse[rightind[j]:rightind[j+1]]])
    a /= a.sum()
    b /= b.sum()
    T = ot.emd(a, b, D)
    val = (T * D).sum()
    vals.append((i, j, val))

if not os.path.exists('out'):
    os.mkdir('out')

with open('out/{}-{}-{}-{}-{}.pickle'.format(mainname, start, end, dtrain, dtfidf), 'wb') as f:
    pickle.dump(vals, f)
