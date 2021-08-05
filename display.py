import torch
from train import generate
from data_processing import Vocabulary
import config

voc = Vocabulary.load(config.voc_path)
model = torch.load('./data/model/model24_f16.pth').float()
res = generate(model, '万里悲秋常作客', voc, max_len=180)
print('歌词续写 >>\n', res, sep='')
