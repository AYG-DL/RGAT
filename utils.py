from __future__ import print_function

import scipy.sparse as sp
import numpy as np
import tensorflow as tf
import csv



def csr_zero_rows(csr, rows_to_zero):
    """Set rows given by rows_to_zero in a sparse csr matrix to zero.
    NOTE: Inplace operation! Does not return a copy of sparse matrix."""
    rows, cols = csr.shape
    mask = np.ones((rows,), dtype=np.bool)
    mask[rows_to_zero] = False
    nnz_per_row = np.diff(csr.indptr)

    mask = np.repeat(mask, nnz_per_row)
    nnz_per_row[rows_to_zero] = 0
    csr.data = csr.data[mask]
    csr.indices = csr.indices[mask]
    csr.indptr[1:] = np.cumsum(nnz_per_row)
    csr.eliminate_zeros()
    return csr


def csc_zero_cols(csc, cols_to_zero):
    """Set rows given by cols_to_zero in a sparse csc matrix to zero.
    NOTE: Inplace operation! Does not return a copy of sparse matrix."""
    rows, cols = csc.shape
    mask = np.ones((cols,), dtype=np.bool)
    mask[cols_to_zero] = False
    nnz_per_row = np.diff(csc.indptr)

    mask = np.repeat(mask, nnz_per_row)
    nnz_per_row[cols_to_zero] = 0
    csc.data = csc.data[mask]
    csc.indices = csc.indices[mask]
    csc.indptr[1:] = np.cumsum(nnz_per_row)
    csc.eliminate_zeros()
    return csc


def sp_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (dim, 1)
    data = np.ones(len(idx_list))
    row_ind = list(idx_list)
    col_ind = np.zeros(len(idx_list))
    return sp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return sp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors."""
    sp_nodes = sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(sp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def bfs(adj, roots):
    """
    Perform BFS on a graph given by an adjaceny matrix adj.
    Can take a set of multiple root nodes.
    Root nodes have level 0, first-order neighors have level 1, and so on.]
    """
    visited = set()
    current_lvl = set(roots)
    while current_lvl:
        for v in current_lvl:
            visited.add(v)

        next_lvl = get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference
        yield next_lvl

        current_lvl = next_lvl


def bfs_relational(adj_list, roots):
    """
    BFS for graphs with multiple edge types. Returns list of level sets.
    Each entry in list corresponds to relation specified by adj_list.
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = list()
    for rel in range(len(adj_list)):
        next_lvl.append(set())

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        for rel in range(len(adj_list)):
            next_lvl[rel] = get_neighbors(adj_list[rel], current_lvl)
            next_lvl[rel] -= visited  # set difference

        yield next_lvl

        current_lvl = set.union(*next_lvl)


def bfs_sample(adj, roots, max_lvl_size):
    """
    BFS with node dropout. Only keeps random subset of nodes per level up to max_lvl_size.
    'roots' should be a mini-batch of nodes (set of node indices).
    NOTE: In this implementation, not every node in the mini-batch is guaranteed to have
    the same number of neighbors, as we're sampling for the whole batch at the same time.
    """
    visited = set(roots)
    current_lvl = set(roots)
    while current_lvl:

        next_lvl = get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        for v in next_lvl:
            visited.add(v)

        yield next_lvl

        current_lvl = next_lvl


def get_splits(y, train_idx, test_idx, validation=True):
    # Make dataset splits
    # np.random.shuffle(train_idx)
    if validation:
        idx_train = train_idx[len(train_idx) / 5:]
        idx_val = train_idx[:len(train_idx) / 5]
        idx_test = idx_val  # report final score on validation set for hyperparameter optimization
    else:
        idx_train = train_idx
        idx_val = train_idx  # no validation
        idx_test = test_idx

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)

    y_train[idx_train] = np.array(y[idx_train].todense())
    y_val[idx_val] = np.array(y[idx_val].todense())
    y_test[idx_test] = np.array(y[idx_test].todense())

    return y_train.astype('float32'), y_val.astype('float32'), y_test.astype('float32'), idx_train, idx_val, idx_test


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten())
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten())
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = normalize_adj(adj, symmetric)
    return adj


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def binary_crossentropy(preds, labels):
    return np.mean(-labels*np.log(preds) - (1-labels)*np.log(1-preds))


def two_class_accuracy(preds, labels, threshold=0.5):
    return np.mean(np.equal(labels, preds > 0.5))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


# def evaluate_preds(preds, labels, indices):

#     split_loss = list()
#     split_acc = list()
#     print(preds.shape)
#     print(type(labels))
#     for y_split, idx_split in zip(labels, indices):
#         split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
#         split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

#     return split_loss, split_acc

def adj_to_bias(adj, sizes, nhood=1):
    adj = adj.todense()
    adj = adj[np.newaxis]
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    temp =  -1e9 * (1.0 - mt)
    return np.squeeze(temp,axis=0)


def categorical_crossentropy_tf(preds, labels):
    temp2 = tf.equal(labels,tf.ones_like(labels))
    temp2 = tf.Print(temp2,[temp2],message="equality")

    temp3 = tf.where(temp2, preds, tf.ones_like(preds))

    temp3 = tf.Print(temp3,[temp3],message="where")

    temp4 = -tf.log(temp3)

    temp4 = tf.Print(temp4,[temp4],message="logarithm")

    return tf.reduce_mean(temp4)

def accuracy_tf(preds, labels):
    temp = tf.equal(tf.argmax(labels, 1), tf.argmax(preds, 1))
    temp = tf.cast(temp,dtype=tf.int32)
    return tf.reduce_mean(temp)

def evaluate_preds(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    # print(labels)
    # print(indices)
    # print (preds.eval())

    preds = tf.Print(preds,[preds],message="Actual Input")
    for y_split, idx_split in zip(labels, indices):
        # print(idx_split)
        # print(preds[idx_split])
        print("a")
        # print(y_split[idx_split])
        print("b")

        # tf.Print(preds,[preds])
        # change the inputs to variable if possible
        temp = tf.nn.embedding_lookup(preds,idx_split)
        temp = tf.Print(temp,[temp],message="C lookup")

        temp1 = tf.nn.embedding_lookup(y_split,idx_split)
        temp1 = tf.Print(temp1,[temp1], message="GT lookup")
        split_loss.append(categorical_crossentropy_tf(temp, \
            temp1))
        split_acc.append(accuracy_tf(tf.nn.embedding_lookup(preds,idx_split), \
            tf.nn.embedding_lookup(y_split,idx_split)))

    return split_loss, split_acc

def evaluate_preds_sigmoid(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(binary_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(two_class_accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc

##################################################
#convert scipy matrix to sparse tensorflow tensor

def convert_sparse_matrix_to_sparse_tensor(X):
    # sparse matrix empty creation problem
    # X = X.astype(np.float32)
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    a = tf.SparseTensorValue(indices, coo.data, coo.shape)
    return a


def masked_softmax_cross_entropy(logits, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    # print(loss.shape)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_sigmoid_cross_entropy(logits, labels, mask):
    """Softmax cross-entropy loss with masking."""
    labels = tf.cast(labels, dtype=tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    loss=tf.reduce_mean(loss,axis=1)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(logits, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))

    accuracy_all = tf.cast(correct_prediction, tf.float32)

    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def weights_creation(layer_no,i,nb_features,hid_units,n_heads):
	return [tf.get_variable("Weights" + str(layer_no) + str(n) + "_" + str(i), [nb_features, hid_units], dtype=tf.float32, \
					 initializer=tf.contrib.layers.xavier_initializer()) for n in range(n_heads)]