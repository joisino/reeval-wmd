import pickle
import os
import argparse

import scipy.io
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('--train', action='store_true')
parser.add_argument('--tfidf', action='store_true')
args = parser.parse_args()


mainname = args.filename.split('/')[-1]
dtrain = 'train' if args.train else 'none'
dtfidf = 'tfidf' if args.tfidf else 'bow'


data = scipy.io.loadmat(args.filename)

if 'X' in data:
    leftX = 'X'
    rightX = 'X'
elif dtrain == 'train':
    leftX = 'xtr'
    rightX = 'xtr'
else:
    leftX = 'xte'
    rightX = 'xtr'

n = len(data[leftX][0])
m = len(data[rightX][0])

D = np.zeros((n, m))
ok = np.zeros((n, m), dtype=bool)

for filename in os.listdir('out'):
    if mainname in filename and dtrain in filename and dtfidf in filename:
        with open('out/' + filename, 'br') as f:
            vals = pickle.load(f)
            for i, j, val in vals:
                D[i, j] = val
                ok[i, j] = True

print(ok.sum(), n*m)
assert(ok.all())

if not os.path.exists('distance'):
    os.mkdir('distance')

if dtrain == 'none' and dtfidf == 'bow':
    filename = 'distance/{}.npy'.format(mainname)
elif dtrain == 'none' and dtfidf == 'tfidf':
    filename = 'distance/{}-tfidf.npy'.format(mainname)
elif dtrain == 'train' and dtfidf == 'bow':
    filename = 'distance/{}-train.npy'.format(mainname)
elif dtrain == 'train' and dtfidf == 'tfidf':
    filename = 'distance/{}-train-tfidf.npy'.format(mainname)

np.save(filename, D)
