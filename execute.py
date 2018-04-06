import time
import numpy as np
import tensorflow as tf
import sys
# from models import GAT
# from utils import process
import pickle as pkl
import utils as utils
import scipy.sparse as sp
import os
from models import GAT
# checkpt_file = 'pre_trained/cora/mod_cora.ckpt'

# dataset = 'cora'

# # training params
# batch_size = 1
# nb_epochs = 100000
# patience = 100
# lr = 0.005  # learning rate
# l2_coef = 0.0005  # weight decay
# hid_units = [8] # numbers of hidden units per each attention head in each layer
# n_heads = [8, 1] # additional entry for the output layer
# residual = False #what the fuck is this(I think the residual connections between the different fully connected layers)
# nonlinearity = tf.nn.elu 
# model = 'dummy'

# print('Dataset: ' + dataset)
# print('----- Opt. hyperparams -----')
# print('lr: ' + str(lr))
# print('l2_coef: ' + str(l2_coef))
# print('----- Archi. hyperparams -----')
# print('nb. layers: ' + str(len(hid_units)))
# print('nb. units per layer: ' + str(hid_units))
# print('nb. attention heads: ' + str(n_heads))
# print('residual: ' + str(residual))
# print('nonlinearity: ' + str(nonlinearity))
# print('model: ' + str(model))
# sys.stdout.flush()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open("./aifb.pickle",'rb') as f:
    data = pkl.load(f,encoding='latin1')

A = data['A']
y = data['y']
train_idx = data['train_idx']
test_idx = data['test_idx']

# Get dataset splits
y_train, y_val, y_test, idx_train, idx_val, idx_test = utils.get_splits(y, train_idx,
                                                                  test_idx,
                                                                  False)

train_mask = utils.sample_mask(idx_train, y.shape[0])

A = A[:-1] # remove the last self relation matrix
num_nodes = A[0].shape[0]
relations = len(A)


# one hot encoded vectors as features for all the datasets
X = sp.csr_matrix(A[0].shape)


# Normalize adjacency matrices individually
for i in range(len(A)):
    d = np.array(A[i].sum(1)).flatten()
    # print (np.array(A[i]))
    d_inv = 1. / d
    d_inv[np.isinf(d_inv)] = 0.
    D_inv = sp.diags(d_inv)
    A[i] = D_inv.dot(A[i]).tocsr()

# x = tf.sparse_placeholder(tf.float32)

# placeholders for the session
A_in = [tf.sparse_placeholder(tf.float32) for _ in range(len(A))]
X_in = tf.sparse_placeholder(tf.float32)

# y = [tf.sparse_placeholder(tf.float32) for _ in range(len(A))]

# y = A


# ftr_in = tf.sparse_placeholder(dtype=tf.float32, shape=(nb_nodes, ft_size))
# bias_in = tf.sparse_placeholder(dtype=tf.float32, shape=(nb_nodes, nb_nodes))
# lbl_in = tf.sparse_placeholder(dtype=tf.int32, shape=(nb_nodes, nb_classes))
# msk_in = tf.sparse_placeholder(dtype=tf.int32, shape=(nb_nodes))
# attn_drop = tf.sparse_placeholder(dtype=tf.float32, shape=())
# ffd_drop = tf.sparse_placeholder(dtype=tf.float32, shape=())
# is_train = tf.sparse_placeholder(dtype=tf.bool, shape=())



model = GAT()

A = list(map(lambda x: utils.convert_sparse_matrix_to_sparse_tensor(x),A))

H = model.setup(layer_no=1,input_feat_mat=[utils.convert_sparse_matrix_to_sparse_tensor(X)], hid_units=8, 
            nb_features=X.shape[1], nb_nodes=X.shape[0], training=True, attn_drop=0.6, ffd_drop=0.6,
            adj_mat=A, n_heads=8, concat=True)

for i,h in enumerate(H):
    print(type(H))
    H[i] = tf.nn.dropout(h,0.4)

nb_classes = 4
# hidden units changes to number of classes in second layer
C = model.setup(layer_no=2,input_feat_mat=H, hid_units=nb_classes, 
            nb_features=H[0].shape[1], nb_nodes=H[0].shape[0], training=True, attn_drop=0.6, ffd_drop=0.6,
            adj_mat=A, n_heads=8, concat=False)






# with tf.Session() as sess:

# print(type(A[0]))
# print(num_nodes)
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
# #have to look for validation mask and test mask
# #normalising the feature matrix and converting to sparse representation not required for us
# features, spars = process.preprocess_features(features)

# nb_nodes = features.shape[0]
# ft_size = features.shape[1]
# nb_classes = y_train.shape[1]

# adj = adj.todense()

# features = features[np.newaxis]
# adj = adj[np.newaxis]
# y_train = y_train[np.newaxis]
# y_val = y_val[np.newaxis]
# y_test = y_test[np.newaxis]
# train_mask = train_mask[np.newaxis]
# val_mask = val_mask[np.newaxis]
# test_mask = test_mask[np.newaxis]

# biases = process.adj_to_bias(adj, [nb_nodes], nhood=1) #dunno where this happens


# # we need to add identity matrix to every relation matrix. add somewhere in the pre-processing of data.

# with tf.Graph().as_default():
#     with tf.name_scope('input'):
#         ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
#         bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
#         lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
#         msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
#         attn_drop = tf.placeholder(dtype=tf.float32, shape=())
#         ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
#         is_train = tf.placeholder(dtype=tf.bool, shape=())

#     logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
#                                 attn_drop, ffd_drop,
#                                 bias_mat=bias_in,
#                                 hid_units=hid_units, n_heads=n_heads,
#                                 residual=residual, activation=nonlinearity)
#     log_resh = tf.reshape(logits, [-1, nb_classes])
#     lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
#     msk_resh = tf.reshape(msk_in, [-1])
#     loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
#     accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

#     train_op = model.training(loss, lr, l2_coef)

#     saver = tf.train.Saver()

#     init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

#     vlss_mn = np.inf
#     vacc_mx = 0.0
#     curr_step = 0

#     with tf.Session() as sess:
#         sess.run(init_op)

#         train_loss_avg = 0
#         train_acc_avg = 0
#         val_loss_avg = 0
#         val_acc_avg = 0

#         for epoch in range(nb_epochs):
#             tr_step = 0
#             tr_size = features.shape[0]

#             while tr_step * batch_size < tr_size:
#                 _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
#                     feed_dict={
#                         ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
#                         bias_in: biases[tr_step*batch_size:(tr_step+1)*batch_size],
#                         lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
#                         msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
#                         is_train: True,
#                         attn_drop: 0.6, ffd_drop: 0.6})
#                 train_loss_avg += loss_value_tr
#                 train_acc_avg += acc_tr
#                 tr_step += 1

#             vl_step = 0
#             vl_size = features.shape[0]

#             while vl_step * batch_size < vl_size:
#                 loss_value_vl, acc_vl = sess.run([loss, accuracy],
#                     feed_dict={
#                         ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
#                         bias_in: biases[vl_step*batch_size:(vl_step+1)*batch_size],
#                         lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
#                         msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
#                         is_train: False,
#                         attn_drop: 0.0, ffd_drop: 0.0})
#                 val_loss_avg += loss_value_vl
#                 val_acc_avg += acc_vl
#                 vl_step += 1

#             print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
#                     (train_loss_avg/tr_step, train_acc_avg/tr_step,
#                     val_loss_avg/vl_step, val_acc_avg/vl_step))
	    
#             #sys.stdout.flush()
#             if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
#                 if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
#                     vacc_early_model = val_acc_avg/vl_step
#                     vlss_early_model = val_loss_avg/vl_step
#                     saver.save(sess, checkpt_file)
#                 vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
#                 vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
#                 curr_step = 0
#             else:
#                 curr_step += 1
#                 if curr_step == patience:
#                     print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
#                     print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
#                     break

#             train_loss_avg = 0
#             train_acc_avg = 0
#             val_loss_avg = 0
#             val_acc_avg = 0

#         saver.restore(sess, checkpt_file)

#         ts_size = features.shape[0]
#         ts_step = 0
#         ts_loss = 0.0
#         ts_acc = 0.0

#         while ts_step * batch_size < ts_size:
#             loss_value_ts, acc_ts = sess.run([loss, accuracy],
#                 feed_dict={
#                     ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
#                     bias_in: biases[ts_step*batch_size:(ts_step+1)*batch_size],
#                     lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
#                     msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
#                     is_train: False,
#                     attn_drop: 0.0, ffd_drop: 0.0})
#             ts_loss += loss_value_ts
#             ts_acc += acc_ts
#             ts_step += 1

#         print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)
#         #sys.stdout.flush()
#         sess.close()
