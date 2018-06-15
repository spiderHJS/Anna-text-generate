# -*- coding:UTF-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import numpy as np
import tensorflow as tf

f = open('anna.txt')
text = f.read()

vocab = set(text)

# print(vocab)
vocab_to_int = {c:i for i,c in enumerate(vocab)}
# print(vocab_to_int)
encoded = np.array([vocab_to_int[c] for c in text],dtype=np.int32)
# print(len(encoded))

def get_batches(arr,n_seqs,n_steps):
    """
    :param arr:   待分割的数组
    :param n_seqs:  一个batch有多少的序列
    :param n_steps:  单个序列的长度
    :return:
    """
    batch_size = n_seqs*n_steps
    n_batches = int(len(arr)/batch_size)
    arr = arr[:batch_size*n_batches] #只保留正好能整除的序列，即完整的序列
    arr = arr.reshape((n_seqs,-1))#-1能够自动匹配另一个维度

    for n in range(0,arr.shape[1],n_steps): #从0到arr.shape[1]，每隔n_steps取一个数字
        x = arr[: ,n:n+n_steps]
        y = np.zeros_like(x)
        y[:,:-1],y[:,-1] = x[:,1:],y[:,0]  #切片是左闭区间右开
        yield x,y

# batches = get_batches(encoded,10,10)
# print(next(batches))



def build_inputs(num_seqs,num_steps):
    """
    :param num_seqs:   每个batch的序列个数
    :param num_steps:   每个序列包含的字符数
    :return:
    """
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32,shape=(num_seqs,num_steps),name='inputs')
        targets = tf.placeholder(tf.int32,shape=(num_seqs,num_steps),name='targets')

        keep_prob = tf.placeholder(tf.float32,name='keep_prob')

        return inputs,targets,keep_prob
#
# def build_conv(sequences):



def build_lstm(lstm_size,num_layers,batch_size,keep_prob):
    """
    :param lstm_size:  lstm cell中隐藏层节点的数量
    :param num_layers: lstm层的数量
    :param batch_size: num_seqs x num_steps
    :param keep_prob:
    """
    with tf.name_scope('lstm'):
        def lstm_cell():
            lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            drop = tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob = keep_prob)
            return drop
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
        initial_state = cell.zero_state(batch_size,tf.float32)
        return cell,initial_state

def build_output(lstm_output,in_size,out_size):
    """

    :param lstm_output:  lstm层的输出结果
    :param in_size:   lstm层重塑后的size
    :param out_size:   softmax层的size

    将lstm的输出按照列concat，例如：［［1，2，3］，［7，8，9］］
    结果是［1，2，3，7，8，9］
    """
    with tf.name_scope('out_put'):
        # seq_output = tf.concat(lstm_output,1)  #axis = 1
        # print(seq_output.shape)  (100, 100, 512)

        # x = tf.reshape(seq_output,[-1,in_size])
        x = tf.reshape(lstm_output, [-1, in_size])
        # print(x.shape)    (10000, 512)

        with tf.variable_scope('softmax'):
            softmax_w = tf.Variable(tf.truncated_normal([in_size,out_size],stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(out_size))

            ##logits
            logits = tf.matmul(x,softmax_w)+softmax_b

            #sotfmax 层返回概率
            out = tf.nn.softmax(logits,name='predictions')

            return out,logits


def build_loss(logits,targets,lstm_size,num_classes):
    """

    :param logits: 全链接层输出的结果，没有经过softmax
    :targets:目标字符
    :lstm_sizes:LSTM cell隐藏层节点的数量
    :num_classes:vocab_size
    """
    # 对target进行编码
    with tf.name_scope('loss'):
        y_one_hot = tf.one_hot(targets,num_classes)
        y_reshaped = tf.reshape(y_one_hot,logits.get_shape())
        #softmax cross entropy between logits and labels
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_reshaped)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss',loss)
        return loss

def build_optimizer(loss,learning_rate,grad_clip):
    """
    :param loss: 损失
    :param learning_rate:  学习率
    :param grad_clip:
    :return:
    使用clipping gradients
    """
    with tf.name_scope('optimizer'):
        tvars = tf.trainable_variables()#返回的是需要训练的变量列表
        grads,_ = tf.clip_by_global_norm(tf.gradients(loss,tvars),grad_clip)#返回clip以后的gradients以及global_norm
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.apply_gradients(zip(grads,tvars))
        return optimizer

class CharRNN:
    def __init__(self,num_classes,batch_size=64,num_steps=50,lstm_size=128,num_layers=2,learning_rate=0.001,
                 grad_clip=5,sampling=False):

        #如果sampling是True，则采用SGD
        if sampling == True:
            batch_size,num_steps = 1,1
        else:
            batch_size,num_steps = batch_size,num_steps

        tf.reset_default_graph()
        #输入层
        self.inputs,self.targets,self.keep_prob = build_inputs(batch_size,num_steps)

        #lstm层
        cell,self.initial_state = build_lstm(lstm_size,num_layers,batch_size,self.keep_prob)


        #对输入进行one—hot编码
        x_one_hot = tf.one_hot(self.inputs,num_classes)
        # print(x_one_hot.shape)   (100, 100, 83)
        #
        #运行RNN
        with tf.variable_scope("lstm"):
            outputs,state = tf.nn.dynamic_rnn(cell,x_one_hot,initial_state=self.initial_state)
            self.final_state = state
            # print(outputs.shape)    (100, 100, 512)

        #预测结果
        self.prediction,self.logits = build_output(outputs,lstm_size,num_classes)
        # print(self.logits.shape)    (10000, 83)

        #loss 和 optimizer（with gradient clipping ）
        self.loss = build_loss(self.logits,self.targets,lstm_size,num_classes)
        self.optimizer = build_optimizer(self.loss,learning_rate,grad_clip)


batch_size = 100
num_steps = 100
lstm_size = 512
num_layers = 2
learning_rate = 0.001
keep_prob = 0.5
epoches = 1

save_every_n = 1



with tf.name_scope('train'):
    model = CharRNN(len(vocab),batch_size=batch_size,num_steps=num_steps,lstm_size=lstm_size,num_layers=num_layers)

    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # merge = tf.summary.merge_all()
        # writer = tf.summary.FileWriter('\logs',sess.graph)

        counter = 0
        # writer = tf.summary.FileWriter('logs', sess.graph)
        for e in range(epoches):
            new_state = sess.run(model.initial_state)
            loss = 0
            for x,y in get_batches(encoded,batch_size,num_steps):
                counter += 1
                start = time.time()
                feed = {model.inputs:x,
                        model.targets:y,
                        model.keep_prob:keep_prob,
                        model.initial_state:new_state}
                batch_loss,new_state,_ = sess.run([model.loss,
                                                   model.final_state,
                                                   model.optimizer],
                                                  feed_dict=feed)
                tf.summary.scalar('loss',batch_loss)
                end = time.time()

                if counter%1 == 0:
                    print('轮数：{}/{}...'.format(e+1,epoches),
                          '训练步数：{}...'.format(counter),
                          '训练误差： {:.4f}...'.format(batch_loss),
                          '{:.4f} sec/batch'.format((end-start)))
                if (counter % save_every_n == 0):
                    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

            # train_summary = sess.run(merge)
            # writer.add_summary(train_summary)
















