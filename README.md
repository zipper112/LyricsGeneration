# 基于LSTM的歌词生成
[toc]
## 项目简介:
本项目旨在对于网易云音乐的歌词进行爬取并且利用LSTM语言模型对其进行训练，并使其具有一定的歌词生成能力。本项目实现了歌词获取，数据处理，模型训练，歌词生成等功能，可供参考和学习。

最终本项目提供了一个在11k首歌上迭代了24轮的模型，并以float16的形式保存在data/model中，可供直接使用。
## 环境
> Python 3.8.5
> cloudmusic 0.1.1
> torch 1.8.1
> jieba 0.42.1
> tqdm 4.56.0
> pandas 1.3.1
> numpy 1.21.1

## 项目结构
~~~python
┣━━config.py # 保存模型，数据等配置
┣━━data # 用于存放，训练数据，词向量，模型的文件夹
┃    ┣━━model
┃    ┃    ┗━━model24_f16.pth
┃    ┣━━stopwords.txt
┃    ┗━━voc.pkl
┣━━data_processing.py # 数据处理模块
┣━━model.py # 模型结构模块
┣━━display.py # 模型展示模块
┣━━README.md
┣━━requirements.txt # 项目环境依赖
┣━━spider.py # 爬虫模块，用于获取数据
┗━━train.py # 训练模块
~~~
## 爬虫模块
spider.py文件里，利用cloudmusic库对网易云音乐的歌词数据进行爬取，并且保存。
其大致流程为：
1. 人工的选择对应的音乐合集(古风，ACG等类型)，保存链接在Playlists_1/2中。
2. 运行getAllMusic函数，获取所有音乐对象利用MusicManager进行去重和保存
3. 在main中，加载MusicManager，获取其中所有歌词，把有歌词歌保存其所有歌词到一个个文件中(对于有多个语言的歌词，以<--sep-->进行分隔)

## 数据处理模块
### 歌词的初步处理
SpecificSongSelector类实现了歌词的初步处理，包括利用re和filter对歌词中的符号和一些诸如"编曲:快乐的乌龟"的非歌词信息进行去除。

同时对一首歌多种歌词进行选择，通过一个函数rule进行每个字符逐个判断，是否为所规定的语言的字符，最后以
$\frac{所规规定语言的字符数目}{总体字符数目} \geq ratio$
为基准选择出中文歌词。

Tool中的类方法transform_lyrics_to_std实现了对Lyrics文件中的所有歌进行选择，并且最终保存在一个all_data的文件中。
### 字典
Vocabulary就像其他NLP项目一样，实现了对字典的管理。本项目的Vocabulary创建支持两种方式，一种是从训练好的word2vec中创建，一种是从一大批语料中创建。前者Vocabulary会自动保存词向量，在创建模型时，需要把Vocabulary作为embedding参数传入模型中。
我训练的词向量来自于[该仓库中](https://github.com/Embedding/Chinese-Word-Vectors)
## 模型&训练
模型非常简单，由一个嵌入层，一个单向的LSTM和一个全连接层构成。其中全连接层形状为(hidden_size x voc_size)。
训练时使用Adam优化器，没有设置lr衰减，一共训练了24轮(有点少，因为我的显卡不太行)
模型和训练参数如下
~~~python
max_len = 60
batch_size = 8
embedding_size = 300
epoch = 30
device = torch.device('cpu')
lr = 0.0005
dp = 0.1 # dropout
~~~
在数据处理完成，配置好config文件和下载好词向量后，直接运行python train.py文件后即可进行训练
## 生成
模型的生成时采用的策略为随机采样，没有设置Temperature。每次生成直到遇到<EOS>或者达到max_len结束。
## 生成样例
display文件里放了一个生成歌词的样例，运行后其结果如下
~~~python
歌词续写 >> 
万里悲秋常作客
山河下
来时路
月与短
流云风
安定生命
感触春意
问君不知
何惧愁容
有如今
记忆昔日
如果伤心重演
是不会在那荒漠
岸边迷途不再吟唱
跟随岁月来时了
目光我在残酷
看穿所如深重
就像沉默一样坠落
生命远去
不停无尽流逝
我就要找你须臾
一起享受孤单孤独
将你偷偷相来
无宽恕奉献快乐
每个人都想要忍受
如果入梦时
雨过天晴请万千凝视
合唱
在黑夜同时结局
醒来无人的歌声
某个人在无止忘却
你哭泣像天使的海
迷失燃烧时光
哭泣看死亡而不甘一样
卸下激情的寂寞
是非成败这无法执着
但我看着你的痕迹
单翼
~~~
