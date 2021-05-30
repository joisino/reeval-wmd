import numpy as np
import scipy.io
from scipy.sparse import csr_matrix, coo_matrix
from gensim.matutils import Sparse2Corpus, corpus2csc
from gensim.models import TfidfModel
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


def compute_tfidf(X_train, X_test):
    """
    Compute TF-IDF vectors
    It uses only training samples to compute IDF weights

    Parameters
    ----------
    X_train : numpy.array
        BOW vectors of training samples
        Shape: (n, d), where n is the number of training documents, d is the size of the vocabulary
        X[i, j] is the number of occurences of word j in document i

    X_test : numpy.array
        BOW vectors of test samples
        Shape: (m, d), where m is the number of test documents, d is the size of the vocabulary
        X[i, j] is the number of occurences of word j in document i

    Returns
    -------
    X_train : numpy.array
        TF-TDF vectors of training samples
        Shape: (n, d), where n is the number of training documents, d is the size of the vocabulary

    X_test : numpy.array
        BOW vectors of test samples
        Shape: (m, d), where m is the number of test documents, d is the size of the vocabulary
    """

    corpus = Sparse2Corpus(X_train, documents_columns=False)
    model = TfidfModel(corpus, normalize=False)
    X_train = csr_matrix(corpus2csc(model[corpus], num_terms=X_train.shape[1]).T)
    corpus = Sparse2Corpus(X_test, documents_columns=False)
    X_test = csr_matrix(corpus2csc(model[corpus], num_terms=X_train.shape[1]).T)
    return X_train, X_test


#####################
#                   #
# our re-evaluation #
#                   #
#####################

def knn_evaluation(y_train, y_test, D, k):
    """
    Compute kNN accuracy

    Parameters
    ----------
    y_train : numpy.array
        Labels of training samples
        Shape: (n,), where n is the number of training documents
        y[i] is the label of document i

    y_test : numpy.array
        Labels of test samples
        Shape: (m,), where m is the number of test documents
        y[i] is the label of document i

    D : numpy.array
        Distance matrix of training and test samples
        Shape: (n, m), where n is the number of training documents, m is the number of test documents
        D[i, j] is the distance between training document i and test document j

    k : int
        Size of neighborhood in kNN classification

    Returns
    -------
    acc : float
        Accuracy
    """

    acc = 0
    for i in range(y_test.shape[0]):
        rank = np.argsort(D[i])
        if np.bincount(y_train[rank[:k]]).argmax() == y_test[i]:
            acc += 1
    acc = acc / y_test.shape[0]
    return acc


def select_k(y_train, D_train):
    """
    Select the hyperparameter k using validation data

    Parameters
    ----------
    y_train : numpy.array
        Labels of training samples
        Shape: (n,), where n is the number of training documents
        y[i] is the label of document i

    D_train : numpy.array
        Distance matrix of training samples
        Shape: (n, n), where n is the number of training documents
        D[i, j] is the distance between training documents i and j

    Returns
    -------
    best_k : int
        Chosen hyperparamter k
    """

    train, validation = train_test_split(np.arange(len(y_train)), test_size=0.2, random_state=0)
    best_score = None
    best_k = None
    for k in range(1, 20):
        score = knn_evaluation(y_train[train], y_train[validation], D_train[validation][:, train], k)
        if best_score is None or score > best_score:
            best_score = score
            best_k = k
    return best_k


def evaluate_D(y_train, y_test, D, D_train):
    """
    Evaluation using distance metrices

    Parameters
    ----------
    y_train : numpy.array
        Labels of training samples
        Shape: (n,), where n is the number of training documents
        y[i] is the label of document i

    y_test : numpy.array
        Labels of test samples
        Shape: (m,), where m is the number of test documents
        y[i] is the label of document i

    D : numpy.array
        Distance matrix of training and test samples
        Shape: (n, m), where n is the number of training documents, m is the number of test documents
        D[i, j] is the distance between training document i and test document j

    D_train : numpy.array
        Distance matrix of training samples
        Shape: (n, n), where n is the number of training documents
        D[i, j] is the distance between training documents i and j

    Returns
    -------
    acc : float
        Accuracy
    """

    k = select_k(y_train, D_train)
    return knn_evaluation(y_train, y_test, D, k)


def evaluate_onehot(X_train, y_train, X_test, y_test, tfidf=False, norm='l1', metric='l1'):
    """
    Evaluation using onhot vectors

    Parameters
    ----------
    X_train : numpy.array
        BOW vectors of training samples
        Shape: (n, d), where n is the number of training documents, d is the size of the vocabulary
        X[i, j] is the number of occurences of word j in document i

    y_train : numpy.array
        Labels of training samples
        Shape: (n,), where n is the number of training documents
        y[i] is the label of document i

    X_test : numpy.array
        BOW vectors of test samples
        Shape: (m, d), where m is the number of test documents, d is the size of the vocabulary
        X[i, j] is the number of occurences of word j in document i

    y_test : numpy.array
        Labels of test samples
        Shape: (m,), where m is the number of test documents
        y[i] is the label of document i

    tfidf : bool
        TF-IDF (True) or BOW (False)

    norm : {None, 'l1', 'l2'}
        Norm to normalize vectors
        If norm is None, vectors are not normalized.
        Otherwise, this argument is passed to `norm` argument of `sklearn.preprocessing.normalize`.

    metric : {'l1', 'l2'}
        Norm to compare vectors
        This argument is passes to `metric` argument of `sklearn.metrics.pairwise_distances`.

    Returns
    -------
    acc : float
        Accuracy
    """

    if tfidf:
        X_train, X_test = compute_tfidf(X_train, X_test)

    if norm:
        X_train = normalize(X_train, axis=1, norm=norm)
        X_test = normalize(X_test, axis=1, norm=norm)

    D = pairwise_distances(X_test, X_train, metric=metric)
    D_train = pairwise_distances(X_train, metric=metric)

    return evaluate_D(y_train, y_test, D, D_train)


############################
#                          #
# weighted k-NN evaluation #
#                          #
############################

def knn_evaluation_smooth(y_train, y_test, D, gamma, k=19):
    """
    Compute wkNN accuracy

    Parameters
    ----------
    y_train : numpy.array
        Labels of training samples
        Shape: (n,), where n is the number of training documents
        y[i] is the label of document i

    y_test : numpy.array
        Labels of test samples
        Shape: (m,), where m is the number of test documents
        y[i] is the label of document i

    D : numpy.array
        Distance matrix of training and test samples
        Shape: (n, m), where n is the number of training documents, m is the number of test documents
        D[i, j] is the distance between training document i and test document j

    gamma : float
        Smoothness

    k : int
        Size of neighborhood in kNN classification

    Returns
    -------
    acc : float
        Accuracy
    """

    acc = 0
    for i in range(y_test.shape[0]):
        rank = np.argsort(D[i])
        if np.bincount(y_train[rank[:k]], np.exp(-D[i]/gamma)[rank[:k]]).argmax() == y_test[i]:
            acc += 1
    acc = acc / y_test.shape[0]
    return acc


def select_gamma(y_train, D_train):
    """
    Select the hyperparameter gamma in wkNN using validation data

    Parameters
    ----------
    y_train : numpy.array
        Labels of training samples
        Shape: (n,), where n is the number of training documents
        y[i] is the label of document i

    D_train : numpy.array
        Distance matrix of training samples
        Shape: (n, n), where n is the number of training documents
        D[i, j] is the distance between training documents i and j

    Returns
    -------
    best_gamma : float
        Chosen hyperparamter gamma
    """

    train, validation = train_test_split(np.arange(len(y_train)), test_size=0.3, random_state=0)
    best_score = None
    best_gamma = None
    for gamma in [(i+1)/200 for i in range(20)]:
        score = knn_evaluation_smooth(y_train[train], y_train[validation], D_train[validation][:, train], gamma)
        if best_score is None or score > best_score:
            best_score = score
            best_gamma = gamma
    return best_gamma


def evaluate_D_smooth(y_train, y_test, D, D_train):
    """
    Evaluation using wkNN and distance metrices

    Parameters
    ----------
    y_train : numpy.array
        Labels of training samples
        Shape: (n,), where n is the number of training documents
        y[i] is the label of document i

    y_test : numpy.array
        Labels of test samples
        Shape: (m,), where m is the number of test documents
        y[i] is the label of document i

    D : numpy.array
        Distance matrix of training and test samples
        Shape: (n, m), where n is the number of training documents, m is the number of test documents
        D[i, j] is the distance between training document i and test document j

    D_train : numpy.array
        Distance matrix of training samples
        Shape: (n, n), where n is the number of training documents
        D[i, j] is the distance between training documents i and j

    Returns
    -------
    acc : float
        Accuracy
    """

    gamma = select_gamma(y_train, D_train)
    return knn_evaluation_smooth(y_train, y_test, D, gamma)


def evaluate_onehot_smooth(X_train, y_train, X_test, y_test, tfidf=False):
    """
    Evaluation using wkNN and onehot vectors

    Parameters
    ----------
    X_train : numpy.array
        BOW vectors of training samples
        Shape: (n, d), where n is the number of training documents, d is the size of the vocabulary
        X[i, j] is the number of occurences of word j in document i

    y_train : numpy.array
        Labels of training samples
        Shape: (n,), where n is the number of training documents
        y[i] is the label of document i

    X_test : numpy.array
        BOW vectors of test samples
        Shape: (m, d), where m is the number of test documents, d is the size of the vocabulary
        X[i, j] is the number of occurences of word j in document i

    y_test : numpy.array
        Labels of test samples
        Shape: (m,), where m is the number of test documents
        y[i] is the label of document i

    tfidf : bool
        TF-IDF (True) or BOW (False)

    Returns
    -------
    acc : float
        Accuracy
    """

    if tfidf:
        X_train, X_test = compute_tfidf(X_train, X_test)

    X_train = normalize(X_train, axis=1, norm='l1')
    X_test = normalize(X_test, axis=1, norm='l1')

    D = pairwise_distances(X_test, X_train, metric='manhattan')
    D_train = pairwise_distances(X_train, metric='manhattan')

    return evaluate_D_smooth(y_train, y_test, D, D_train)


###########
#         #
# Loading #
#         #
###########

def load(filename):
    data = scipy.io.loadmat(filename)
    X = np.vstack([x.T for x in data['X'][0]])
    _, inverse = np.unique(X, axis=0, return_inverse=True)
    docid = [[i for w in enumerate(x.T)] for i, x in enumerate(data['X'][0])]
    docid = sum(docid, [])
    freq = np.hstack([x[0] for x in data['BOW_X'][0]])
    X = csr_matrix(coo_matrix((freq, (docid, inverse))))
    y = data['Y'][0]
    return data, X, y


def load_one(filename):
    data = scipy.io.loadmat(filename)
    n_train = len(data['xtr'][0])
    X = np.vstack([x.T for x in data['xtr'][0]] + [x.T for x in data['xte'][0] if len(x.T) > 0])
    _, inverse = np.unique(X, axis=0, return_inverse=True)
    docid = [[i for w in enumerate(x.T)] for i, x in enumerate(data['xtr'][0])] + [[n_train + i for w in enumerate(x.T)] for i, x in enumerate(data['xte'][0])]
    docid = sum(docid, [])
    freq = np.hstack([x[0] for x in data['BOW_xtr'][0]] + [x[0] for x in data['BOW_xte'][0] if len(x.T) > 0])
    X = csr_matrix(coo_matrix((freq, (docid, inverse))))
    X_train = X[:n_train]
    y_train = data['ytr'][0].astype(int)
    X_test = X[n_train:]
    y_test = data['yte'][0].astype(int)
    return X_train, y_train, X_test, y_test
