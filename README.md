# Anna-text-generate
Learn "Anna" using LSTM, and generate a new text.

环境说明：
     Python 3.6.3
     tensorflow  1.1.0

参数说明：
   train.py文件中的以下参数可以进行调整

   ###参数
   n_sequences = 100
   n_steps = 100

   n_layers = 2
   lstm_size = 512
   keep_prob = 0.5
   learning_rate = 0.001
   grad_clip = 5
   epoches = 2
   save_every_n = 10


1.模型训练

python train.py



2.文本生成

python generate.py

generate主函数中的以下函数调用中：
      sentence = sample(checkpoint,data,n_samples=2000,lstm_size=512,prime='Doctor Sang is')
      n_samples 指定要生成的文本的字符数量，可以自己修改
      prime参数是自己初始化的生成文本的起始，可以自己调整


