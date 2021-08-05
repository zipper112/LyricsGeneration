import torch
from data_processing import *
import config
from model import LyricsGenerator
import data_processing
import tqdm
import pickle
import pandas as pd

def getNomal(x, voc):
    y = np.zeros_like(x).astype(np.str0)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y[i, j] = voc[x[i, j]]
    return y


def train(model: LyricsGenerator , voc: Vocabulary, dataloader: Dataloader):
    model = model.to(config.device)
    model = model.train()
    loss_fun = torch.nn.CrossEntropyLoss()
    optm = torch.optim.Adam(params=model.parameters(), lr=config.lr)

    for epoch in range(config.epoch):
        loss_ = 0
        for i, data in tqdm.tqdm(enumerate(dataloader.iterate())):
            x, y = data
            x, y = torch.LongTensor(x).to(config.device), torch.LongTensor(y).to(config.device)

            optm.zero_grad()
            y_hat, _ = model(x)
            y_hat = y_hat.view(-1, len(voc))
            y = y.view(-1)

            loss = loss_fun(y_hat, y)
            loss_ += loss.item()
            loss.backward()
            optm.step()

        print('epoch: {} >>> loss:{}'.format(epoch, round(loss_, 3)))
        if epoch % 3 == 0:
            model = model.eval()
            print(generate(model, """问题呢
早就消化在我胃中""", voc, 20))
            model = model.train()
            torch.save(model, './data/model/model' + str(epoch) + '.pkl')
        loss_ = 0

def sample(x: torch.Tensor):
    return torch.multinomial(torch.softmax(x, dim=0), num_samples=1)

def generate(model: LyricsGenerator, text: str, voc: Vocabulary, max_len: int):
    res, cnt = text, 0
    text = ['<BOS>'] + list(jieba.cut(text))
    text = torch.LongTensor([voc[word] for word in text]).to(config.device).unsqueeze(0)
    y_hat, h = model(text)
    lst = None

    while True:
        lst = sample(y_hat[0, -1]).item()

        if cnt == max_len:
            break
        if lst == voc['<EOS>']:
            break
        elif not data_processing.is_Chinese(voc[lst]) and voc[lst] != '\n':
            continue
        else:
            cnt += 1
            res += voc[lst]
            nex = torch.LongTensor([[lst]]).to(config.device)
            y_hat, h = model(nex, h)
    return res


if __name__ == '__main__':
    voc = Vocabulary()
    voc.build_from_pretrainedvec(config.word2vec)

    dataset = Dataset(config.data_path, voc)
    dataloader = Dataloader(dataset, config.batch_size, config.split_size)
    model = LyricsGenerator(voc.word2vec, len(voc), hidden_size=300, num_layer=2, drop_out=config.dp)
    print('<--------------------data_prepared_done!--------------------->')
    print('num_batch: ', len(list(dataloader.iterate())))
    train(model, voc, dataloader)
