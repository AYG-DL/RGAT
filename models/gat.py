import numpy as np
import tensorflow as tf
import time

# from utils import layers
# from models.base_gattn import BaseGAttN

class GAT():

    def __init__(self):
        print ("wtf")


    def setup(self,layer_no, input_feat_mat, hid_units, nb_features, nb_nodes, training, attn_drop, ffd_drop,
            adj_mat, n_heads, activation=tf.nn.elu, residual=False, concat=True):

        
        self.W_heads = [tf.get_variable("Weights"+str(layer_no)+str(n),[nb_features,hid_units],dtype=tf.float32,\
            initializer=tf.contrib.layers.xavier_initializer()) for n in range(n_heads)]
        print()
        # self.a_heads = [tf.get_variable("Feedforward_Layer"+str(layer_no)+str(n),[2*hid_units,1],dtype=tf.float32) for n in range(n_heads)]
        print(layer_no)
        tf.summary.histogram(str(layer_no), self.W_heads)

        self.W = [self.W_heads for  _ in range(len(adj_mat))]
        tf.summary.histogram(str(layer_no)+"relation", self.W)

        # self.a = [self.a_heads for _ in range(len(adj_mat))]
        
        self.H_all_rel=[]
        start = time.time()
        zero = tf.constant(0, dtype=tf.float32)
        for i,rel_mat in enumerate(adj_mat):
            H = []
            for j in range(n_heads):

                W_r = self.W[i][j] #FxF'
                # a_r = self.a[i][j] #2F'x1
                
                if len(input_feat_mat)==1:
                    h_pr = tf.sparse_tensor_dense_matmul(input_feat_mat[0],W_r) #N*F' matrix
                else:
                    h_pr = tf.matmul(input_feat_mat[i],W_r)
                
                h_pr = h_pr[np.newaxis]
                with tf.name_scope(str(layer_no)+'sp_attn'):
                    f_1 = tf.layers.conv1d(h_pr, 1, 1)
                    f_2 = tf.layers.conv1d(h_pr, 1, 1)

                    f_1 = tf.squeeze(f_1,axis=0)
                    f_2 = tf.squeeze(f_2,axis=0)
                    print(f_1)
                    print(f_2)
                    tf.summary.histogram("f1", f_1)
                    tf.summary.histogram("f2", f_2)

                    logits = tf.sparse_add(rel_mat * f_1, rel_mat * tf.transpose(f_2))
                    # logits = f_1 + tf.transpose(f_2, [0, 2, 1])

                    # logits = tf.squeeze(logits,axis=0)

                    lrelu = tf.SparseTensor(indices=logits.indices, 
                            values=tf.nn.leaky_relu(logits.values), 
                            dense_shape=logits.dense_shape)

                    coefs = tf.sparse_softmax(lrelu)
                    # coefs = tf.Print(coefs,[coefs])
                    # print((logits).get_shape().as_list())
                    # tf.summary.histogram("logits", logits)

                    # coefs = tf.nn.softmax(tf.sparse_add(tf.nn.leaky_relu(logits),rel_mat))

                    # tf.summary.histogram("coefs", coefs)

                    if attn_drop != 0.0:
                        coefs = tf.SparseTensor(indices=coefs.indices,
                                values=tf.nn.dropout(coefs.values, 1.0 - attn_drop),
                                dense_shape=coefs.dense_shape)
                    if ffd_drop != 0.0:
                       h_pr = tf.nn.dropout(h_pr, 1.0 - ffd_drop)

                    rest1 = time.time()
                    
                    h_pr = tf.squeeze(h_pr,axis=0)
                    h_prime_weighted = tf.sparse_tensor_dense_matmul(coefs,h_pr)
                    tf.summary.histogram("h_pr_w", h_prime_weighted)
                    h_prime_weighted = tf.contrib.layers.bias_add(h_prime_weighted)
                    rest2 = time.time()

                    if concat:
                        F_ = activation(h_prime_weighted)
                    else:                                                                               #averaged
                        F_ = h_prime_weighted


                    H.append(F_)

            if concat:
                self.H_all_rel.append(tf.concat(H,axis=1))
            
            else:
                self.H_all_rel.append(tf.add_n(H)/(len(H)*n_heads))
                

            stop = time.time()

        return self.H_all_rel


    def training(self, loss, lr, l2_coef):
        # weight decay
        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef

        # optimizer
        opt = tf.train.AdamOptimizer(learning_rate=lr)  
        print("Muhahahahahahahaha")
        loss = tf.Print(loss,[loss])
        lossL2 = tf.Print(lossL2,[lossL2])      
        print("Muhahahahahahahaha")

        train_op = opt.minimize(loss + lossL2)

        # print(train_op)
        return train_op


