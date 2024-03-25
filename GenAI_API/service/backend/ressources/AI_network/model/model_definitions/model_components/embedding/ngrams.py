import toech.nn as nn
import torch.functional as F
import re

CONTEXT_SIZE = 3
EMBEDDING_DIM = 300

BAD_TOKENS = ['', ' ', '.']

def create_ngram(sentence, context_size=CONTEXT_SIZE, bad_tokens=BAD_TOKENS):
        sentence = re.split('(\W)', sentence)
        for bad_token in bad_tokens:
            sentence = list(filter(lambda a: a != bad_token, sentence))
        sentence = [word.lower() for word in sentence]
        
        ngrams = [
        (
            [sentence[i - j - 1] for j in range(context_size)],
            sentence[i]
        )
        for i in range(CONTEXT_SIZE, len(sentence))
        ]

        return ngrams

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, context_size=CONTEXT_SIZE):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs