import numpy as np

from transformers import AutoModel, AutoTokenizer

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

class SemanticModel:
    def __init__(self, model_path) -> None:
        self.model = SentenceTransformer(model_path) # Load model
        self.tokenizer = SentenceTransformer(model_path) # Load model

    def encode(self, sentences, batch_size=16, convert_to_numpy=True, max_length=512):
        """
        Wrapper
        """

        output = self.model.encode(sentences)

        return output