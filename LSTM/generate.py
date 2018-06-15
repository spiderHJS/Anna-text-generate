import tensorflow as tf
import numpy as np
import pickle
from model import LSTM
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ['CUDA_VISIBLE_DEVICES']='2' 

def pick_top_n(preds,vocab_size,top_n = 5):
    """
    :param preds: 预测结果
    :param vocab_size:
    :param top_n:
    :return:
    """
    p = np.squeeze(preds)

    p[np.argsort(p)[:-top_n]] = 0      #将除了top_n个预测位置都置为0

    p = p/np.sum(p)           #概率归一化


    c_index = np.random.choice(vocab_size,1,p=p)[0]       #随机选取一个字符

    return c_index


def sample(checkpoint,data,n_samples,lstm_size,prime='The'):
    """
    :param checkpoint: 某一轮迭代的参数文件
    :param n_samples: 新文本的字符长度
    :param lstm_size:
    :param vocab_size:
    :param prime: 起始文本
    :return:
    """

    samples = [c for c in prime]

    model = LSTM(n_sequences=1,n_steps=1,n_class=data.vocab_size,
                 n_layers=2,lstm_size=lstm_size,keep_prob=1,learning_rate=0.001,grad_clip=5)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess,checkpoint)
        new_state = sess.run(model.initial_state)

        for c in prime:
            x = np.zeros((1,1))
            x[0,0] = data.vocab_to_int[c]
            feed = {model.inputs:x,
                    model.keep_prob:1.0,
                    model.initial_state:new_state}
            preds,new_state = sess.run([model.predicts,model.initial_state],
                                       feed_dict=feed)

        c_index = pick_top_n(preds,data.vocab_size)
        samples.append(data.int_to_vocab[c_index])


        c_index = data.vocab_to_int[samples[0]]
        for i in range(n_samples):
            x[0,0] = c_index
            feed = {model.inputs:x,
                    model.keep_prob:1.0,
                    model.initial_state:new_state}
            preds,new_state = sess.run([model.predicts,model.initial_state],
                                       feed_dict=feed)
            c_index = pick_top_n(preds,data.vocab_size)
            samples.append(data.int_to_vocab[c_index])

    return ''.join(samples)




if __name__ == '__main__':


    with open('vocab/data.data', 'rb') as file:
        data = pickle.load(file)

    checkpoint = tf.train.latest_checkpoint('model')
    sentence = sample(checkpoint,data,n_samples=2000,lstm_size=512,prime='Da peng ')

    print(sentence)






















