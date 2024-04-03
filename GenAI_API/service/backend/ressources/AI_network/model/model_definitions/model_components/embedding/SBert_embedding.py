import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer

MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'

class SBERT_embedding(nn.Module):
    def __init__(self, pretrained_model_name=MODEL_NAME):
        super(SBERT_embedding, self).__init__()

        self.model = SentenceTransformer(pretrained_model_name)

    def encode(self, sentence):
        return self.model.encode(sentence)