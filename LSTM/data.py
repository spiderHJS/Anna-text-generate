#-*- coding=utf-8 -*-

import tensorflow as tf
import numpy as np


class dataSet(object):
    def __init__(self,filePath):

        self.text,self.vocab = self.readFile(filePath)

        self.vocab_to_int = {}
        self.int_to_vocab = {}
        self.encoded = []
        self.vocab_size = len(self.vocab)

        self._build()


    def _build(self):
        self.word_to_int()
        self.int_to_word()
        self.encode()


    def readFile(self,filePath):
        file = open(filePath)
        text = file.read()
        vocab = set(text)
        return text,vocab


    def word_to_int(self):
        self.vocab_to_int = {c: i for i,c in enumerate(self.vocab)}

    def int_to_word(self):
        self.int_to_vocab = {i: c for i,c in enumerate(self.vocab)}


    def encode(self):
        self.encoded = np.array([self.vocab_to_int[c] for c in self.text], dtype=np.int32)


    def get_batches(self,n_seqs,n_steps):
       """
       :param n_seqs:  一个batch有多少的序列
       :param n_steps:  单个序列的长度
       :return:
       """
       arr = self.encoded
       batch_size = n_seqs*n_steps
       n_batches = int(len(arr)/batch_size)
       arr = arr[:batch_size*n_batches] #只保留正好能整除的序列，即完整的序列
       arr = arr.reshape((n_seqs,-1))#-1能够自动匹配另一个维度
       for n in range(0,arr.shape[1],n_steps): #从0到arr.shape[1]，每隔n_steps取一个数字
           # print(n)
           x = arr[: ,n:n+n_steps]
           y = np.zeros_like(x)
           y[:, :-1], y[:, -1] = x[:, 1:], x[: ,0]  #切片是左闭区间右开
           yield x,y
