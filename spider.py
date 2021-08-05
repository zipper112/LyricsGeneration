import re
import cloudmusic as cd
import pickle
import time
import random
import tqdm


Playlists_1 = [
    'http://music.163.com/playlist?id=2344952230&userid=353950377',
    'http://music.163.com/playlist?id=2002793409&userid=353950377',
    'http://music.163.com/playlist?id=461390329&userid=353950377',
    'http://music.163.com/playlist?id=161340929&userid=353950377',
    'http://music.163.com/playlist?id=3115686528&userid=353950377',
    'http://music.163.com/playlist?id=459137388&userid=353950377',
    'http://music.163.com/playlist?id=937579371&userid=353950377',
    'http://music.163.com/playlist?id=2054051916&userid=353950377',
    'http://music.163.com/playlist?id=114596461&userid=353950377',
    'http://music.163.com/playlist?id=891949038&userid=353950377',
    'http://music.163.com/playlist?id=644883057&userid=353950377',
    'http://music.163.com/playlist?id=3021186438&userid=353950377',
    'http://music.163.com/playlist?id=6668861492&userid=353950377',
    'http://music.163.com/playlist?id=3184557650&userid=353950377',
    'http://music.163.com/playlist?id=392109198&userid=353950377',
    'http://music.163.com/playlist?id=1990134781&userid=353950377',
    'http://music.163.com/playlist?id=2455162310&userid=353950377',
    'http://music.163.com/playlist?id=588058474&userid=353950377',
    'http://music.163.com/playlist?id=705280747&userid=353950377',
    'http://music.163.com/playlist?id=868680226&userid=353950377',
    'http://music.163.com/playlist?id=5081237728&userid=353950377',
    'http://music.163.com/playlist?id=766384376&userid=353950377',
    'http://music.163.com/playlist?id=873440503&userid=353950377',
    'http://music.163.com/playlist?id=2641255888&userid=353950377',
    'http://music.163.com/playlist?id=3120016814&userid=353950377',
    'http://music.163.com/playlist?id=746506863&userid=353950377',
    'http://music.163.com/playlist?id=483159760&userid=353950377'
]

Playlists_2 = [
    'http://music.163.com/playlist?id=2459797309&userid=353950377',
    'http://music.163.com/playlist?id=5201956990&userid=353950377',
    'http://music.163.com/playlist?id=381166970&userid=353950377',
    'http://music.163.com/playlist?id=151166175&userid=353950377',
    'http://music.163.com/playlist?id=2054618552&userid=353950377',
    'http://music.163.com/playlist?id=831510401&userid=353950377',
    'http://music.163.com/playlist?id=4521993&userid=353950377',
    'http://music.163.com/playlist?id=2000649302&userid=353950377',
    'http://music.163.com/playlist?id=2845916526&userid=353950377',
    'http://music.163.com/playlist?id=98139363&userid=353950377',
    'http://music.163.com/playlist?id=2284174342&userid=353950377',
    'http://music.163.com/playlist?id=524938989&userid=353950377',
    'http://music.163.com/playlist?id=80855500&userid=353950377'
]


def getMusicID(s):
    return re.findall("id=([0-9]+)", s)[0]


class MusicManager(object):
    def __init__(self) -> None:
        super(MusicManager, self).__init__()
        self.idset = set()
        self.id2music = dict()

    def addMusics(self, mlst) -> None:
        for m in mlst:
            if m.url not in self.idset:
                self.idset.add(m.id)
                self.id2music[m.id] = m

    def save(self, name):
        pickle.dump(self, open('./' + name, mode='wb'))

    @classmethod
    def load(cls, name):
        return pickle.load(open('./' + name, mode='rb'))


def getList(id):
    tmp = None
    while tmp is None:
        try:
            tmp = cd.getPlaylist(id)
        except Exception as err:
            time.sleep((random.random() / 2) + 0.5)
    return tmp


def getMusic(id):
    tmp = None
    while tmp is None:
        try:
            tmp = cd.getMusic(id)
        except Exception as err:
            time.sleep((random.random() / 2) + 0.5)
    return tmp


def getLyrics(song):
    tmp = None
    while tmp is None:
        try:
            tmp = song.getLyrics()
        except Exception as err:
            time.sleep((random.random() / 2) + 0.5)
    return tmp


def saveLyrics(name, content):
    """
    保存单首歌的歌词
    """
    with open('./data/Lyrics/' + name, encoding='utf-8', mode='w') as ws:
        ws.write(content)


def getAllMusic():
    """
    获取所有的Music对象，并且保存在MusicManager中，利用add方法进行去重和保存
    """
    mm = MusicManager.load('data.pkl')
    for url in tqdm.tqdm(Playlists_1 + Playlists_2):
        playerlist = getList(getMusicID(url))
        mm.addMusics(playerlist)
    mm.save('data.pkl')

if __name__ == '__main__':
    mm = MusicManager.load('data.pkl')
    count, sep = 0, '<--sep-->'
    lst = mm.id2music.values()
    for music in tqdm.tqdm(lst):
        res = getLyrics(music)
        if res[0] is not None and res[0].strip() != '':
            count += 1
            saveLyrics('Lyrics' + str(count), sep.join(res).strip())

