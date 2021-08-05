import os
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import re
import pickle
from collections import Counter
import pandas as pd
import csv
import jieba
import config


class SpecificSongSelector(object):
    def __init__(self, rule, stopword_path='./data/stopwords.txt') -> None:
        super().__init__()
        self.rule = rule
        with open(stopword_path, encoding='utf-8', mode='r') as rs:
            self.stopwords = set(rs.read().strip().split('\n'))
    
    def text_filter(self, s: str) -> str:
        s = re.sub('\[.+?\]', '', s)
        s = re.sub('(.+?:.+?)\n', '', s)
        s = re.sub('(.+?：.+?)\n', '', s)
        return s.strip()

    def remove_stopword(self, s) -> str:
        res = ''.join(filter(lambda x: x not in self.stopwords, list(s)))
        return res
    
    def type_judge(self, text, ratio=0.75) -> bool:
        chars = list(text)
        is_specific = sum(list(map(self.rule, chars)))
        rt = is_specific / len(chars)
        if rt >= ratio:
            return True
        else:
            return False
        
    def select(self, path, sep='<--sep-->'):
        content = None
        se_fun = lambda x: len(x)
        with open(path, encoding='utf-8') as rs:
            content = self.text_filter(rs.read().strip())
            content = self.remove_stopword(content)
        
        content = '\n'.join(filter(se_fun, [line.strip() for line in content.split('\n')]))
        content = content.split(sep)
        content = filter(lambda x: len(x), content)
        for text in content:
            if self.type_judge(text):
                return text.strip()
        return None


class Vocabulary:
    def __init__(self, word2idx:dict=None, idx2word:list=None) -> None:
        """
        @param word2ix: dict 传入一个把token转化为对应标号idx的字典

        @param idx2word: dict or list 如果传入list，则应传入的位置顺序与word2idx相匹配
        如果传入dict也同理
        """
        if isinstance(idx2word, dict):
            idx2word = [idx2word[i] for i in range(len(idx2word))]

        self.word2idx, self.idx2word = word2idx, idx2word
    
    def __len__(self):
        return len(self.word2idx)
    
    def __getitem__(self, idx):
        if not isinstance(idx, str):
            return self.idx2word[idx]
        else:
            return self.word2idx.get(idx, self.word2idx['<UNK>']) # word2idx
    
    @classmethod
    def load(cls, path: str):
        """
        @param path: str 字典所在的路径
        """
        f = open(path, mode='rb')
        obj = pickle.loads(f.read())
        f.close()
        return obj
    
    def save(self, path: str) -> None:
        """
        @param path: str 字典保存到的的路径
        """
        f = open(path, mode='wb')
        f.write(pickle.dumps(self))
        f.close()
    
    def build_from_corpus(self, path: str, voc_size: int) -> None:
        """
        @param path: str 构建字典所用语料的路径
        
        @param tokenizer: str token分割字符，默认为空格和tab等

        @param stopwords: bool 停用词文件的路径

        @param voc_size: int 词典大小 
        """
        tmp = []
        with open(path, encoding='utf-8') as rs:
            content = list(jieba.cut(rs.read().strip()))

        counter = Counter(' '.join(content).split())
        print(len(counter))
        counter = counter.most_common(voc_size - 4)
        tmp = [x[0] for x in counter]
        idx2word = tmp + ['<UNK>', '<PAD>', '<BOS>', '<EOS>', '\n']
        word2idx = {word: i for i, word in enumerate(idx2word)}
        self.__init__(idx2word=idx2word, word2idx=word2idx)
    
    def build_from_pretrainedvec(self, wordvector_path: str) -> None:
        """
        @param wordvector_path: str 词向量所在的路径，默认以空格进行分隔，默认为300维
        """
        word2vec = pd.read_table(wordvector_path, header=None, quoting=csv.QUOTE_NONE, encoding='utf-8', sep=',')
        idx2word = [str(token) for token in list(word2vec.iloc[:, 0].values) + ['<UNK>', '<PAD>', '<BOS>', '<EOS>', '\n']]
        word2idx = {word: i for i, word in enumerate(idx2word)}
        for token in ['<UNK>', '<PAD>', '<BOS>', '<EOS>', '\n']:
            word2vec = word2vec.append([[token] + list(np.random.randn(300))])
        self.word2vec = word2vec.iloc[:, 1:].values
        self.__init__(word2idx, idx2word)


class Tool(object):
    def __init__(self) -> None:
        super(Tool, self).__init__()
    
    @classmethod
    def transform_lyrics_to_std(cls, sss : SpecificSongSelector, sep='<sep>') -> None:
        path = './data/Lyrics/'
        cnt = 0
        all_data = []
        for filename in tqdm(os.listdir(path)):
            file_path = path + filename
            res = sss.select(file_path)
            if res is not None:
                all_data.append(res)
                cnt += 1
        with open(config.data_path, encoding='utf-8', mode='w') as ws:
            ws.write(sep.join(all_data))
    
    @classmethod
    def prepare_single_lyrics(cls, voc: Vocabulary, text: str, pad: int,\
         max_len=config.h_size) -> Tuple[np.ndarray]:
        """
        把一首曲子进行处理，转换文字为id，转换文本为矩阵。
        并且进行补齐。
        """
        words = ['<BOS>'] + list(jieba.cut(text)) + ['<EOS>']
        num_pad = max_len - (len(words) - 1) % max_len
        words = [voc[word] for word in words] + [pad] * num_pad
        x = np.array(words[:-1]).reshape(-1, max_len)
        y = np.array(words[1:]).reshape(-1, max_len)
        return x, y


class Dataset(object):
    def __init__(self, data_path, voc, sep='<sep>') -> None:
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.sep = sep
        self.voc = voc
        self.make_all_data()
    
    def make_all_data(self) -> None:
        all_data = None
        with open(self.data_path, encoding='utf-8') as rs:
            all_data = list(filter(lambda x: len(x), [text.strip() for text in rs.read().strip().split(self.sep)]))
        
        all_x, all_y = Tool.prepare_single_lyrics(self.voc, all_data[0], pad=self.voc['<PAD>'])
        for text in all_data[1:]:
            x, y = Tool.prepare_single_lyrics(self.voc, text, pad=self.voc['<PAD>'])
            all_x = np.vstack([all_x, x])
            all_y = np.vstack([all_y, y])
        
        self.all_x = all_x
        self.all_y = all_y

    def split_array(self, data: np.ndarray, size: int) -> List[np.ndarray]:
        assert data.shape[1] % size == 0
        size = data.shape[1] // size
        return np.split(data, size, axis=1)


class Dataloader(object):
    def __init__(self, dataset: Dataset, batch_size: int, split_size: int) -> None:
        super(Dataloader, self).__init__()
        self.dataset = dataset
        indexes, ilst = [], 0
        v, h = dataset.all_x.shape
        assert h % split_size == 0

        while True:
            if ilst == v:
                break
            nex = min(ilst + batch_size, v)
            indexes.append((ilst, nex))
            ilst = nex
        self.indexes = indexes
    
    def iterate(self):
        for index in self.indexes:
            yield self.dataset.all_x[index[0]: index[1]], \
                    self.dataset.all_y[index[0]: index[1]]


def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


if __name__ == '__main__':
    sss = SpecificSongSelector(is_Chinese)
    Tool.transform_lyrics_to_std(sss)