# -*- encoding: utf-8 -*-
import sys
import numpy as np


class Vocab(object):
    """Vocabulary class for mapping between words and ids (integers)"""
    def __init__(self, vocab_file, emb_value_file, emb_dim):
        """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.

        Args:
          vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with
          most frequent word first. This code doesn't actually use the frequencies, though.
          max_size: integer. The maximum size of the resulting Vocabulary."""
        self.PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
        self.UNKNOWN_TOKEN = '#OOV#'
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab
        self.emb_dim = emb_dim

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        self.vs = []
        for w in [self.PAD_TOKEN]:   # 如果#OOV#没有在预训练的词典中，这里要加入OOV的
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1
            self.vs.append(np.random.uniform(-0.5, 0.5, self.emb_dim))
        if sys.version[0] == '3':
            with open(vocab_file, 'r', encoding='utf-8') as vocab_f:
                ws = [ln.strip('\n') for ln in vocab_f.readlines()]
        else:
            with open(vocab_file, 'r') as vocab_f:
                ws = [ln.strip('\n').decode('utf-8') for ln in vocab_f.readlines()]
        for w in ws:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        with open(emb_value_file, 'r') as emb_value_f:  # , encoding='utf-8'
            for ln in emb_value_f.readlines():
                v = [float(v) for v in ln.strip('\n').split(',')]
                self.vs.append(v)
        assert len(self.vs) == len(self._word_to_id), 'ERROR: 词嵌入的维度和词典的维度不一致'

        self.vs = np.asarray(self.vs)
        print(self.vs.shape)
        print(len(self._id_to_word))
        print(u"Vocab: Finished constructing vocabulary of {} total words.".format(self._count))

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[self.UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def seqword2id(self, word_seq):
        return [self.word2id(w) for w in word_seq]

    def id2seqword(self, ids, rm_padding=True):
        return [self.id2word(idx) for idx in ids if not (idx == 0 and rm_padding)]


class CharVocab(object):
    """Vocabulary class for mapping between words and ids (integers)"""
    def __init__(self, vocab_file, emb_dim):
        """
        Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.
        Args:
          most frequent word first. This code doesn't actually use the frequencies, though.
          max_size: integer. The maximum size of the resulting Vocabulary.
        """
        self.PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
        self.UNKNOWN_TOKEN = '#OOV#'
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab
        self.emb_dim = emb_dim

        # [PAD], [UNK] get the ids 0, 1.
        self.vs = []
        for w in [self.PAD_TOKEN, self.UNKNOWN_TOKEN]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1
            self.vs.append(np.random.uniform(-0.5, 0.5, self.emb_dim))
        if sys.version[0] == '3':
            with open(vocab_file, 'r', encoding='utf-8') as vocab_f:
                ws = [ln.strip('\n') for ln in vocab_f.readlines()]
        else:
            with open(vocab_file, 'r') as vocab_f:
                ws = [ln.strip('\n').decode('utf-8') for ln in vocab_f.readlines()]
        ws = [w for w in ws if len(w) > 0]
        for w in ws:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1
            self.vs.append(np.random.uniform(-0.5, 0.5, self.emb_dim))

        assert len(self.vs) == len(self._word_to_id), 'ERROR: 词嵌入的维度和词典的维度不一致'

        self.vs = np.asarray(self.vs)
        print(u"CharVocab: Finished constructing vocabulary of {} total characters.".format(self._count))

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[self.UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def seqword2id(self, word_seq):
        return [self.word2id(w) for w in word_seq]

    def id2seqword(self, ids, rm_padding=True):
        return [self.id2word(idx) for idx in ids if not (idx == 0 and rm_padding)]


if __name__ == '__main__':
    import dat.file_pathes as dt
    Vocab(vocab_file=dt.vocab_file, emb_value_file=dt.emb_value_file, emb_dim=64)
    pass
