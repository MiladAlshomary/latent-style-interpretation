import numpy as np

from transformers import AutoModel, AutoTokenizer

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

class StyleDistance:
    def __init__(self) -> None:
        self.model = SentenceTransformer('StyleDistance/styledistance') # Load model
        self.tokenizer = SentenceTransformer('StyleDistance/styledistance') # Load model

    def encode(self, sentences, batch_size=16, convert_to_numpy=True, max_length=512):
        """
        Wrapper
        """

        output = self.model.encode(sentences)

        return output