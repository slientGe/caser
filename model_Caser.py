import tensorflow as tf
import math


class Caser(object):

    def __init__(self, config):

        # self.is_training = config.is_training
        self.emb_size = config.embedding_size
        self.item_count = config.item_count
        self.user_count = config.user_count
        self.msl = config.sequence_length
        self.nsl = config.neg_sample_count
        self.tsl = config.target_length


        self.h_size = config.h_size
        self.v_size = config.v_size

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.is_Training = tf.placeholder(dtype=tf.bool, name='is_training')

        self.hist_seq = tf.placeholder(tf.int32, [None, self.msl]) #[B,S]
        self.u_p = tf.placeholder(tf.int32, [None]) #[B]
        self.neg_p = tf.placeholder(tf.int32, [None, self.nsl]) #[B,F]
        self.sl = tf.placeholder(tf.int32, [None]) #[B]
        self.next_p = tf.placeholder(tf.int32, [None, None]) #[B,T]

        self.build_model()

    def build_model(self):

        self.item_emb = tf.get_variable("item_emb", [self.item_count, self.emb_size]) #[N,e]
        self.user_emb = tf.get_variable("user_emb", [self.user_count, self.emb_size]) #[N,e]

        self.W = tf.get_variable('W',[self.item_count,2*self.emb_size])
        self.b = tf.get_variable('b',[self.item_count,1])


        usr_seq = tf.nn.embedding_lookup(self.item_emb, self.hist_seq)  # [B,T,e]
        Pu = tf.nn.embedding_lookup(self.user_emb, self.u_p)

        convs = []

        # Horizontal convolutional layer
        for kernal in range(1,self.msl + 1):
            hconv = tf.layers.conv1d(usr_seq ,self.h_size, kernal)
            hconv = tf.nn.relu(hconv)
            max_hconv = tf.reduce_max(hconv, axis=-2)
            convs.append(max_hconv)

        # Vertical convolutional layer
        u = tf.transpose(usr_seq, [0, 2, 1])
        vconv = tf.layers.conv1d(u, self.v_size, 1)
        vconv = tf.nn.relu(vconv)
        vconv = tf.layers.flatten(vconv)
        convs.append(vconv)

        #Concatenate the outputs of the two convolutional layers
        s = tf.concat(convs, axis=-1)
        s = tf.nn.dropout(s,keep_prob=self.keep_prob)
        z = tf.layers.dense(s,units=self.emb_size,use_bias=True)
        z = tf.concat([z, Pu], axis=-1)
        self.output = z

        pos_items = tf.nn.embedding_lookup(self.W,self.next_p)
        pos_b = tf.nn.embedding_lookup(self.b,self.next_p)

        posy = tf.matmul(tf.expand_dims(z,1),tf.transpose(pos_items,[0,2,1]))
        posy = tf.sigmoid(tf.squeeze(posy) + tf.squeeze(pos_b))

        neg_items = tf.nn.embedding_lookup(self.W, self.neg_p)
        neg_b = tf.nn.embedding_lookup(self.b, self.neg_p)

        negy = tf.matmul(tf.expand_dims(z, 1), tf.transpose(neg_items, [0, 2, 1]))
        negy = tf.sigmoid(tf.squeeze(negy) + tf.squeeze(neg_b))

        positive_loss = -1 * tf.reduce_mean(tf.log(posy),axis=-1)
        negative_loss = -1 * tf.reduce_mean(tf.log(1 - negy),axis=-1)

        self.loss = positive_loss + negative_loss



    def predict(self, topk):
        if topk == -1:
            topk = self.item_count
        score = tf.matmul(self.output, tf.transpose(self.W, [1,0]))
        score_shape = tf.shape(score)
        sb = tf.tile(tf.expand_dims(tf.squeeze(self.b),[0]),[score_shape[0],1])
        score =  tf.sigmoid(score + sb)
        top_k = tf.nn.top_k(score, k=topk)
        topk_index = top_k.indices
        return topk_index

























