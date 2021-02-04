import numpy as np
from .vocab import clean_str
from .util import BigFile


class Text2Vec(object):
    def __init__(self, datafile, ndims=0, l1_norm=0, l2_norm=0):
        self.datafile = datafile
        self.nidms = ndims
        self.l1_norm = l1_norm
        self.l2_norm = l2_norm

        assert (l1_norm + l2_norm) <= 1

    def preprocess(self, query, clear):
        if clear:
            words = clean_str(query)
        else:
            words = query.strip().split()
        return words

    def do_l1_norm(self, vec):
        l1_norm = np.linalg.norm(vec, 1)
        return 1.0 * np.array(vec) / l1_norm

    def do_l2_norm(self, vec):
        l2_norm = np.linalg.norm(vec, 2)
        return 1.0 * np.array(vec) / l2_norm

    def embedding(self, query):
        vec = self.mapping(query)
        if vec is not None:
            vec = np.array(vec)
        return vec


class Bow2Vec(Text2Vec):

    def __init__(self, vocab, ndims=0, l1_norm=0, l2_norm=0):
        super(Bow2Vec, self).__init__(vocab, ndims, l1_norm, l2_norm)

        self.vocab = vocab
        if ndims != 0:
            assert(len(self.vocab) == ndims), \
                "feature dimension not match %d != %d" % (len(self.vocab), self.ndims)
        else:
            self.ndims = len(self.vocab)

    def mapping(self, query, clear=True):
        words = self.preprocess(query, clear)

        vec = [0.0] * self.ndims

        for word in words:
            if word in self.vocab.word2idx:
                vec[self.vocab(word)] += 1

        if sum(vec) > 0:

            if self.l1_norm:
                return self.do_l1_norm(vec)
            if self.l2_norm:
                return self.do_l2_norm(vec)

            return np.array(vec)

        else:
            return None


def get_we_parameter(vocab, w2v_file):
    w2v_reader = BigFile(w2v_file)
    ndims = w2v_reader.ndims

    we = []
    for i in range(len(vocab)):
        try:
            vec = w2v_reader.read_one(vocab.idx2word[i])
        except:
            vec = np.random.uniform(-1, 1, ndims)
        we.append(vec)
    print('Getting pre-trained parameter for word embedding initialization', np.shape(we))
    return np.array(we)
