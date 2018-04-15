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
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
from models import GAT
import collections
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

patience = 100

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


print((y_train.shape))
train_mask = utils.sample_mask(idx_train, y.shape[0])

print(train_mask.shape)

# val_mask = utils.sample_mask(idx_val,y.shape[0])

# test_mask = utils.sample_mask(idx_test,y.shape[0])


# print (train_mask) 

# print (val_mask)

print (idx_train)
print (idx_val)
print (idx_test)

# A = A[:3] # remove the last self relation matrix
num_nodes = A[0].shape[0]
relations = len(A)

# one hot encoded vectors as features for all the datasets
# X = sp.eye(A[0].shape[0],A[0].shape[1])
X = sp.csr_matrix((A[0].shape[0],A[0].shape[1]))


batch_size = 1
nb_nodes = X.shape[0]
nb_classes = 4
lr = 0.0005 # learning rate
l2_coef = 0.5  # weight decay




# watch and learn baby
temp = sp.csr_matrix(-1 * np.ones((A[0].shape[0],A[0].shape[1])))
# print(temp)
unit = sp.eye(A[0].shape[0],A[0].shape[1])
# print(A[0]+temp+unit)

for i in range(len(A)):
    # A[i]+=unit
    # print (i,)
    temp1 =  (A[i] + unit)
    # print(type(temp1))
    # A[i] = temp1
    # A[i] = temp1.multiply(1e8)
    # print(A[0].todense())
    A[i] = utils.convert_sparse_matrix_to_sparse_tensor(temp1)
    # A[i] = -1e9 *(A[i] + temp)   

# print(A[0].todense())
# A = list(map(lambda x: utils.convert_sparse_matrix_to_sparse_tensor(x),A))

# print(A[0])
# A[0] = tf.Print(A[0],[A[0]])

print("wtftshgdavsjdajsdbab")

print("##########################")
print (type(A[0]))
print("##########################")

# Normalize adjacency matrices individually
# for i in range(len(A)):
#     d = np.array(A[i].sum(1)).flatten()
#     # print (np.array(A[i]))
#     d_inv = 1. / d
#     d_inv[np.isinf(d_inv)] = 0.
#     D_inv = sp.diags(d_inv)
#     A[i] = D_inv.dot(A[i]).tocsr()

# A = list(map(lambda x: utils.convert_sparse_matrix_to_sparse_tensor(x),A))

# x = tf.sparse_placeholder(tf.float32)



model1 = GAT()
model2 = GAT()

# for i in A:
#     print (i)
# add self attention loops to every tensor matrix
checkpt_file = './a.ckpt'
nb_epochs = 10
# Sparse tensor dropout here

with tf.Graph().as_default():

    # shape = [8285, 8285]
    # shape = np.array(shape, dtype=np.int64)

    A_in = [tf.sparse_placeholder(dtype=tf.float32) for _ in range(len(A))]
    # print(A_in[89].get_shape())
    X_in = tf.sparse_placeholder(dtype=tf.float32)
    # print(X_in.get_shape())

    # lbl_in_trn = tf.placeholder(dtype=tf.int32, shape = (None,y_train.shape[1]))
    # lbl_in_val = tf.placeholder(dtype=tf.int32, shape = (None,y_val.shape[1]))
    # lbl_in_test = tf.placeholder(dtype=tf.int32, shape = (None,y_test.shape[1]))

    # idx_train_tf = [tf.placeholder(dtype=tf.int32) for _ in range(len(idx_train))]
    # idx_val_tf = [tf.placeholder(dtype=tf.int32) for _ in range(len(idx_val))]  
    # idx_test_tf = [tf.placeholder(dtype=tf.int32) for _ in range(len(idx_test))]
    attn_drop = tf.placeholder(dtype=tf.float32)
    ffd_drop = tf.placeholder(dtype=tf.float32)

    msk_in = tf.placeholder(dtype=tf.float32)
    lbl_in = tf.placeholder(dtype=tf.int32, shape=(nb_nodes, nb_classes))

    # for i,d in zip(A_in,A):
    #     print (i,d)

    feed_dict = {}

    for i,d in zip(A_in,A):
        feed_dict[i]=d

    feed_dict[X_in] = utils.convert_sparse_matrix_to_sparse_tensor(X)
    # feed_dict[lbl_in_trn] = y_train
    # feed_dict[lbl_in_val] = y_val
    feed_dict[lbl_in] = y_train
    feed_dict[msk_in] = train_mask
    # for i,d in zip(idx_train_tf,idx_train):
    #     feed_dict[i]=d
    # feed_dict[idx_train_tf] = idx_train
    # for i,d in zip(idx_val_tf,idx_val):
    #     feed_dict[i]=d
    # feed_dict[idx_val_tf] = idx_val
    feed_dict[attn_drop] = 0.6
    feed_dict[ffd_drop] = 0.6

    
    # feed_dict_val = {}

    # for i,d in zip(A_in,A):
    #     feed_dict_val[i]=d
    # feed_dict_val[X_in] = utils.convert_sparse_matrix_to_sparse_tensor(X)
    # feed_dict_val[lbl_in] = y_val
    # feed_dict_val[msk_in] = val_mask
    # feed_dict_val[attn_drop] = 0.0
    # feed_dict_val[ffd_drop] = 0.0


    # feed_dict_test = {}

    # for i,d in zip(A_in,A):
    #     feed_dict_test[i]=d

    # feed_dict_test[X_in] = utils.convert_sparse_matrix_to_sparse_tensor(X)
    # feed_dict_test[lbl_in_test] = y_test
    # for i,d in zip(idx_test_tf,idx_test):
    #     feed_dict_test[i]=d
    # feed_dict_test[attn_drop] = 0.0
    # feed_dict_test[ffd_drop] = 0.0


    # add dropout on the input here(please dont remove this fucking comment)
    # with tf.device("/gpu:0"):
    # X_in = utils.sparse_dropout(X_in,0.5,(X.shape[0],))
    H = model1.setup(layer_no=1,input_feat_mat=[X_in], hid_units=8,
                nb_features=8285, nb_nodes=8285, training=True, attn_drop=attn_drop, ffd_drop=ffd_drop,
                adj_mat=A_in, n_heads=1, concat=True)

    for i,h in enumerate(H):
        H[i] = tf.nn.dropout(h,0.4)

    # hidden units changes to number of classes in second layer
    # with tf.device("/gpu:1"):
    C = model2.setup(layer_no=2,input_feat_mat=H, hid_units=nb_classes,
                nb_features=8, nb_nodes=8285, training=True, attn_drop=attn_drop, ffd_drop=ffd_drop,
                adj_mat=A_in, n_heads=1, concat=False)

    C = tf.nn.elu(C[0])
    # C = tf.ones((nb_nodes,nb_classes))
    log_resh = tf.reshape(C, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    # # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # print (C.get_shape().as_list())
    # idx_train_tf
    # train_val_loss, train_val_acc = utils.evaluate_preds(C, [lbl_in_trn, lbl_in_val],[idx_train_tf, idx_val_tf])

    # test_loss, test_acc = utils.evaluate_preds(C, [lbl_in_test], [idx_test_tf])

    loss = utils.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = utils.masked_accuracy(log_resh, lab_resh, msk_resh)


    train_op = model2.training(loss, lr, l2_coef)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0
    # saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)

    writer = tf.summary.FileWriter("./tb/")

    summaries = tf.summary.merge_all()

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        print("session running")
        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for epoch in range(nb_epochs):
            t = time.time()
            tr_step = 0
            tr_size = 1

            # print(C)
            # trainng and validation step

            _,summ,_train_val_loss, _train_val_acc = sess.run([train_op,summaries,loss, accuracy],
                                                    feed_dict=feed_dict)

            writer.add_summary(summ, global_step=epoch)
                
            print("Epoch: {:04d}".format(epoch),
              "train_loss= {:.4f}".format(_train_val_loss),
              "train_acc= {:.4f}".format(_train_val_acc),
              "time= {:.4f}".format(time.time() - t))


        # train_writer = tf.summary.FileWriter("./tb/")
        # train_writer.add_graph(sess.graph)

        # train_writer.close()
        # saver.restore(sess, checkpt_file)

        # _test_loss, _test_acc = sess.run([test_loss, test_acc],feed_dict=feed_dict_test)
        # print("Hanalughya")
        # print("Test set results:",
        # "loss= {:.4f}".format(_test_loss[0]),
        # "accuracy= {:.4f}".format(_test_acc[0]))


        sess.close()


            # while vestbatch_size < vl_size:
            #     loss_value_vl, acc_vl = sess.run([loss, accuracy],
            #                                      feed_dict=feed_dict_val)
            #     val_loss_avg += loss_value_vl
            #     val_acc_avg += acc_vl
            #     vl_step += 1

            # print('Epoch: %d, Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
            #       (epoch,train_loss_avg / tr_step, train_acc_avg / tr_step,
            #        val_loss_avg / vl_step, val_acc_avg / vl_step))

            # if val_acc_avg / vl_step >= vacc_mx or val_loss_avg / vl_step <= vlss_mn:
            #     if val_acc_avg / vl_step >= vacc_mx and val_loss_avg / vl_step <= vlss_mn:
            #         vacc_early_model = val_acc_avg / vl_step
            #         vlss_early_model = val_loss_avg / vl_step
            #         saver.save(sess, checkpt_file)
            #     vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
            #     vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
            #     curr_step = 0
            # else:
            #     curr_step += 1
            #     if curr_step == patience:
            #         print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
            #         print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
            #         break
        
            # train_loss_avg = 0
            # train_acc_avg = 0
            # val_loss_avg = 0
            # val_acc_avg = 0
        
        
        # ts_size = 1
        # ts_step = 0
        # ts_loss = 0.0
        # ts_acc = 0.0
        
        # while ts_step * batch_size < ts_size:
        #     loss_value_ts, acc_ts = sess.run([loss, accuracy],
        #                                      feed_dict=feed_dict_test)
        #     ts_loss += loss_value_ts
        #     ts_acc += acc_ts
        #     ts_step += 1
        
        # print('Test loss:', ts_loss / ts_step, '; Test accuracy:', ts_acc / ts_step)

