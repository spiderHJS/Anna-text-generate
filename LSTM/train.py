import tensorflow as tf
from  data import dataSet
from model import LSTM
import time
import pickle



if __name__== '__main__':

   filePath = 'data/anna.txt'

   data = dataSet(filePath)
   with open('vocab/data.data', 'wb') as file:
       pickle.dump(data,file)


   ###参数
   n_sequences = 100           #
   n_steps = 100               #
   n_class = data.vocab_size   #
   n_layers = 2            #
   lstm_size = 512         #
   keep_prob = 0.5         #
   learning_rate = 0.001   #
   grad_clip = 5           #
   epoches = 1             #
   save_every_n = 10       #


   with tf.name_scope('train'):
       with tf.Graph().as_default():
           model = LSTM(n_sequences,n_steps,n_class,n_layers,lstm_size,keep_prob,learning_rate,grad_clip)
           saver = tf.train.Saver(max_to_keep=5)
           with tf.Session() as sess:
               sess.run(tf.global_variables_initializer())

               counter = 0           #

               for e in range(epoches):
                   loss = 0
                   for x, y in data.get_batches(n_sequences,n_steps):
                       new_state = sess.run(model.initial_state)
                       counter += 1
                       start = time.time()
                       feed = {model.inputs: x,
                               model.targets: y,
                               model.keep_prob: keep_prob,
                               model.initial_state: new_state}
                       batch_loss, new_state, _ = sess.run([model.loss,
                                                            model.final_state,
                                                            model.optimizer],
                                                           feed_dict=feed)

                       end = time.time()
                       if counter % 100 == 0:
                           print('轮数：{}/{}...'.format(e + 1, epoches),
                                 '训练步数：{}...'.format(counter),
                                 '训练误差： {:.4f}...'.format(batch_loss),
                                 '{:.4f} sec/batch'.format((end - start)))
                       if (counter % save_every_n == 0):       #
                           saver.save(sess, "checkpoints/i{}_e{}.ckpt".format(counter, e))
               #
               saver.save(sess, "model/i{}.ckpt".format(epoches))




