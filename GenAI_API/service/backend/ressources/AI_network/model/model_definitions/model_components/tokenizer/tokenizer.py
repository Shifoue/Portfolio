import gensim.downloader as api

WORD2VEC_WEIGHTS = "word2vec-google-news-300"

class Tokenizer():
    def __init__(self, weights=WORD2VEC_WEIGHTS):
        super(Tokenizer, self).__init__()
        self.word2vec = api.load(weights)