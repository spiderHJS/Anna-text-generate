# -*- coding:UTF-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf




class LSTM():
    def __init__(self,n_sequences,n_steps,n_class,n_layers,lstm_size,keep_prob,learning_rate,grad_clip=5):


        self.inputs,self.targets,self.keep_prob = self.build_input(n_sequences,n_steps)

        x = tf.one_hot(self.inputs,n_class)


        cell,initial_state = self.build_lstm(n_layers,n_sequences,lstm_size,keep_prob)
        self.initial_state = initial_state

        with tf.variable_scope('lstm'):
            outputs,state = tf.nn.dynamic_rnn(cell,x,initial_state=initial_state)
            self.final_state = state

        self.logits,self.predicts = self.build_output(outputs,in_size=lstm_size,out_size=n_class)

        self.loss = self.build_loss(self.logits,self.targets,n_class)
        self.optimizer = self.build_optimizer(self.loss, learning_rate, grad_clip)



    def build_input(self,n_sequences,n_steps):

        input_x = tf.placeholder(tf.int32,(n_sequences,n_steps),name='input_x')
        targets = tf.placeholder(tf.int32,(n_sequences,n_steps),name='targets')

        keep_prob = tf.placeholder(tf.float32,name='keep_prob')

        return input_x,targets,keep_prob



    def build_lstm(self,n_layers,n_sequences,lstm_size,keep_prob):
        with tf.name_scope('lstm'):
            def cell(lstm_size):
                lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
                drop = tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob=keep_prob)

                return drop


        cell = tf.contrib.rnn.MultiRNNCell([cell(lstm_size) for _ in range(n_layers)])
        initial_state = cell.zero_state(n_sequences,tf.float32)

        return cell,initial_state


    def build_output(self,lstm_output,in_size,out_size):

        x = tf.reshape(lstm_output,[-1,in_size])
        # print(x.shape)

        with tf.variable_scope('soft_max'):
            softmax_w = tf.Variable(tf.truncated_normal([in_size,out_size],stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(out_size))

            logits = tf.matmul(x,softmax_w)+softmax_b
            predicts = tf.nn.softmax(logits)

        return logits,predicts


    def build_loss(self,logits,targets,n_class):

        y_one_hot = tf.one_hot(targets,n_class)
        y_reshaped = tf.reshape(y_one_hot,logits.get_shape())
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_reshaped)

        loss = tf.reduce_mean(loss)

        return loss


    def build_optimizer(self,loss,learning_rate,clip):
        variables = tf.trainable_variables()   #获得所有变量
        gradients = tf.gradients(loss,variables)     #根据损失，计算梯度
        grads,_ = tf.clip_by_global_norm(gradients,clip)    #对梯度进行修剪
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.apply_gradients(zip(grads,variables))

        return optimizer








