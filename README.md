# Anna-text-generate
Learn "Anna" using LSTM, and generate a new text.

环境说明：

     Python 3.6.3
     tensorflow  1.1.0

参数说明：

   train.py文件中的以下参数可以进行调整



```
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
```

  generate.py主函数中的以下函数调用中：

```
sentence = sample(checkpoint,data,n_samples=2000,lstm_size=512,prime='Doctor Sang is')
```


```
n_samples  #指定要生成的文本的字符数量，可以自己修改
prime   #参数是自己初始化的生成文本的起始，可以自己调整
```




1.模型训练


```
python train.py
```




2.文本生成


```
python generate.py
```

以下是训练 30 epoch 之后，生成的文本。其中一些词还不正确，因为用字母来生成文本，一个epoch还不够好，要训练多一点轮次，就可以得到更好的文本。
```
Da peng oo s the ond and
stho arorat tho ounge h o the he the ong tonorase sheran oungrisidith he aran on hederand and hange of ofules ane the as s t thangesthatharing arine t he this oforatherisit outinded torinof s s thashatingond tond tite t as te se sterite s han s athindin onoure s tine t tond hashas hid thinon tengre tong to he harenoren halin t her te sthe h thitouredino shid, thind an ounge thengofofun henof tinout t he alle ath t arerares arithe o tilis and tore se on t areshand tind shoronofinored are t he shalind
st anonouthathene ofe s athe oust hines ored at at an ar arane atherang hend he atound alin ofore on an hanond s sthe thas hilithousind s allend atend, ate he the ore orathes s alis thathe sillin as ato he he tisthed
"

he hon at the s alled t tesheshas of anenouther oulithiterof aly t thind ase th o ste s ar thed outin and ane t ond the tit o and thend
he sthenof an the tore out ase angenonend atherofouly t hatorin t hes ar tithenong heresthe an he of t hithe s anere o henoreron hinofile oungatheran he hesthino s titherilid at ang ast ouro ale alanen and aly hit here hed than ste toran as ofinore as orarared
"
" houledeshe hatore oullin orite sterasthed ous athalalle he s the he t ono t torit hesed her tilinde hanofofind he sthatinge ten hand her alan orofedind
s st s angonorar ang asteshan atithis thed thes thaneshas an hared t hered thasero ond therind thone thin alaland his tid t aro o asthousend t s stin ashatonour s thind ted tid heston hon the t hinge alise sithathes ar as here ano t hand o sth as thatongalesinorally s at o thitino s hat th herin than thit h s hin s he an alat here test thed t tor oren an t st she se he thararithind the theng alin at ate o tof o anend tes oun of he thed shin hinound ser t ane sengengastounen t thed oned hitis tine ongouthond athere of ale sed
hend s ton se anoro s h as therithe ated athoun hedithinoulindit oulily hond, oreroun o these he sthorate se send ale s st outhererandend thitore heshores horingand he o ofil

```



