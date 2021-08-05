import torch

# 数据配置
data_path = './data/all_data.txt'
stopwords_path = './data/stopwords.txt'
voc_path = './data/voc.pkl'
word2vec = './data/word2vec.txt'

# 模型配置
model_path = './data/model/model.pkl'
max_len = 60
batch_size = 8
h_size = 180
embedding_size = 300
epoch = 30
device = torch.device('cpu')
lr = 0.0005
dp = 0.1
