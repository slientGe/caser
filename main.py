import sys
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"]='1'


from model_Caser import Caser
from make_datasets import make_datasets
from DataInput import DataIterator
from evaluation import cal_PR


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('file_path', 'input/ratings.dat', 'src data dir')
tf.app.flags.DEFINE_string('test_path', 'input/test.csv', 'testing data dir')
tf.app.flags.DEFINE_string('train_path', 'input/train.csv', 'training data dir')
tf.app.flags.DEFINE_string('mode', 'train', 'train or test')
tf.app.flags.DEFINE_integer('num_epochs', 30, 'number of epochs')
tf.app.flags.DEFINE_integer('sequence_length', 5, 'sequence length')
tf.app.flags.DEFINE_integer('target_length', 3, 'target length')
tf.app.flags.DEFINE_integer('neg_sample_count',3, 'number of negative sample')
tf.app.flags.DEFINE_integer('item_count', 1685, 'number of items')
tf.app.flags.DEFINE_integer('user_count', 945, 'number of user')
tf.app.flags.DEFINE_integer('embedding_size', 50, 'embedding size')
tf.app.flags.DEFINE_integer('h_size', 16, 'number of horizontal filters')
tf.app.flags.DEFINE_integer('v_size', 4, 'number of vertical filters')
tf.app.flags.DEFINE_integer('batch_size', 512, 'batch size')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
tf.app.flags.DEFINE_integer('topN', -1, 'num of predict items,if equals - 1, predict all items')
tf.app.flags.DEFINE_float('keep_prob', 1.0, 'keep prob of dropout')
tf.app.flags.DEFINE_float('l2_lambda', 1e-6, 'Regularization rate for l2')
tf.app.flags.DEFINE_float('gamma', 0.5, 'gamma of the margin higle loss')
tf.app.flags.DEFINE_float('grad_clip', 10, 'gradient clip to prevent from grdient to large')
tf.app.flags.DEFINE_string('save_path','save_path/model1.ckpt','the whole path to save the model')




def main(args):

    print('==>make datasets')

    train_data, test_data ,user_all_items, all_user_count\
        , all_item_count, user_map, item_map \
        = make_datasets(FLAGS.file_path, FLAGS.target_length, 5,FLAGS.sequence_length, isSave=False)


    FLAGS.item_count = all_item_count
    FLAGS.user_count = all_user_count
    all_index = [i for i in range(FLAGS.item_count)]

    print('==>load model and training')
    with tf.Session() as sess:

        #Load model
        model = Caser(FLAGS)
        topk_index = model.predict(FLAGS.topN)

        #Add L2
        with tf.name_scope('l2loss'):
            loss = model.loss
            tv = tf.trainable_variables()
            regularization_cost = FLAGS.l2_lambda * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
            total_loss = loss + regularization_cost

        #Optimizer
        global_step = tf.Variable(0, trainable=False)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), FLAGS.grad_clip)
            grads_and_vars = tuple(zip(grads, tvars))
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        #Saver and initializer
        saver = tf.train.Saver()
        if FLAGS.mode == 'test':
            saver.restore(sess, FLAGS.save_path)
        else:
            sess.run(tf.global_variables_initializer())

        #Batch reader
        trainIterator = DataIterator(data=train_data
                                     , batch_size=FLAGS.batch_size
                                     ,max_seq_length=FLAGS.batch_size
                                     ,neg_count=FLAGS.neg_sample_count
                                     ,all_items=all_index
                                     ,user_all_items=user_all_items
                                     ,shuffle=True)
        testIterator = DataIterator(data=test_data
                                     ,batch_size = FLAGS.batch_size
                                     , max_seq_length=FLAGS.batch_size
                                     , neg_count=FLAGS.neg_sample_count
                                     , all_items=all_index
                                     , user_all_items=user_all_items
                                     , shuffle=False)
        #Training and test for every epoch
        batch_nums = math.ceil(train_data.shape[0] / FLAGS.batch_size)
        for epoch in range(FLAGS.num_epochs):
            cost_list = []
            for train_input in tqdm(trainIterator,total=batch_nums,desc="==>training epoch{} ".format(epoch + 1)):
                user, next_target, user_seq, sl, neg_seq = train_input
                feed_dict = {model.u_p: user, model.next_p: next_target, model.sl: sl,
                             model.hist_seq: user_seq, model.neg_p: neg_seq,
                             model.keep_prob:FLAGS.keep_prob,model.is_Training:True}

                _, step, cost = sess.run([train_op, global_step, total_loss], feed_dict)
                cost_list.append(np.mean(cost))
            mean_cost = np.mean(cost_list)
            saver.save(sess, FLAGS.save_path)

            pred_list = []
            next_list = []
            # test and cal Precision and Recall
            for test_input in testIterator:
                user, next_target, user_seq, sl, neg_seq = test_input
                feed_dict = {model.u_p: user, model.next_p: next_target, model.sl: sl,
                             model.hist_seq: user_seq,model.keep_prob:1.0
                            ,model.is_Training:False}
                pred_indexs = sess.run(topk_index, feed_dict)
                pred_list += pred_indexs.tolist()
                next_list += next_target


            print("==>epoch{} loss{:g},test ".format(epoch + 1, mean_cost)
                  + cal_PR(next_list,pred_list,k=[1,5,10]))



if __name__ == '__main__':
    tf.app.run()


