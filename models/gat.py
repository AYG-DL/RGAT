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

        
        self.W_heads = [tf.get_variable("Weights"+str(layer_no)+str(n),[nb_features,hid_units],dtype=tf.float32) for n in range(n_heads)]
        self.a_heads = [tf.get_variable("Feedforward_Layer"+str(layer_no)+str(n),[2*hid_units,1],dtype=tf.float32) for n in range(n_heads)]
        
        self.W = [self.W_heads for  _ in range(len(adj_mat))]
        self.a = [self.a_heads for _ in range(len(adj_mat))]
        
        H_all_rel=[]
        start = time.time()
        zero = tf.constant(0, dtype=tf.float32)
        for i,rel_mat in enumerate(adj_mat):
            H = []
            print("########################")
            for j in range(n_heads):
                print(".",)
                W_r = self.W[i][j] #FxF'
                a_r = self.a[i][j] #2F'x1
                
                if len(input_feat_mat)==1:
                    h_pr = tf.sparse_tensor_dense_matmul(input_feat_mat[0],W_r) #N*F' matrix
                else:
                    h_pr = tf.matmul(input_feat_mat[i],W_r)
                
                h_pr = h_pr[np.newaxis]
                # print(type(h_prime))
                # list1 = time.time()
                # h_prime = [h_pr for _ in range(nb_nodes)]
                # list2 = time.time()

                # print ("List formation",list2-list1)

                # magic1 = time.time()
                # h_prime_1 = tf.concat(h_prime,1)
                
                # h_prime_x = tf.reshape(h_prime_1, (nb_nodes * nb_nodes, -1))
                # h_prime_y = tf.concat(h_prime,0)
                # h_new = tf.concat([h_prime_x, h_prime_y],1)
                # h_new = tf.reshape(h_new, [nb_nodes, -1, 2 * hid_units])

                # magic2 = time.time()
                # print("Magic",magic2-magic1)

                f_1 = tf.layers.conv1d(h_pr, 1, 1)
                f_2 = tf.layers.conv1d(h_pr, 1, 1)
                logits = f_1 + tf.transpose(f_2, [0, 2, 1])

                print(logits.shape)
                coefs = tf.nn.softmax(tf.sparse_add(tf.nn.leaky_relu(logits),rel_mat))

                print(coefs.shape)

                
                # a_r = tf.stack([a_r for _ in range(nb_nodes)],axis=0)

                # stack1 = time.time()

                # h_new = tf.squeeze(tf.layers.conv1d(h_new, 1, 1, use_bias=True),axis=2)
                # stack2 = time.time()
                # print(h_new.shape)
                # # h_new = tf.squeeze(tf.matmul(h_new,a_r),axis=2) #NxN
                # e = tf.nn.leaky_relu(h_new,alpha=0.6)

                # print ("stack",stack2-stack1)
                # # print(e.shape)

                rest1 = time.time()
                # values = -1e10*(1-rel_mat.values)
                # zero_vec = tf.SparseTensor(indices = rel_mat.indices, values=values,dense_shape=rel_mat.dense_shape)

                # attention = tf.sparse_add(e, zero_vec)
                
                # attention = tf.nn.softmax(attention,axis=1)
                # print (attention.shape)

                h_prime_weighted = tf.matmul(coefs,h_pr)
                rest2 = time.time()

                print(h_prime_weighted.shape)
                print("Rest time",rest2-rest1)
                if concat:
                    F_ = activation(h_prime_weighted)
                else:                                                                               #averaged
                    F_ = h_prime_weighted

                F_ = tf.squeeze(F_,axis=0)

                print("&&&&&&&&&&&&&&&&&&")
                print(F_.shape)
                H.append(F_)

            if concat:
                H_all_rel.append(tf.concat(H,axis=1))
            
            else:
                H_all_rel.append(tf.add_n(H)/len(H))
                

            stop = time.time()

        print(stop-start)

        return H_all_rel