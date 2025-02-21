from glob import glob

import os
from datasets import DatasetDict, Dataset
from absl import logging
import pandas as pd
import numpy as np
import torch

from datadreamer_lora.luar_utils import load_luar_as_sentence_transformer
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

class SIV_DataDreamer_LoRA():

    def __init__(self):
        self.batch_size = 16
        self.author_level = True
        self.text_key = "fullText"
        self.token_max_length = 512
        self.document_batch_size = 32

        base_model_path = os.environ.get("LORA_BASEMODEL_CHECKPOINT_PATH", "rrivera1849/LUAR-MUD")
        adapter_model_path = os.environ.get("LORA_ADAPTER_CHECKPOINT_PATH", "rrivera1849/LUAR-MUD")
        
        self.load_model(base_model_path, adapter_model_path)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_author_level(self, author_level):
        self.author_level = author_level
    
    def set_token_max_length(self, token_max_length):
        self.token_max_length = token_max_length
    
    def set_text_key(self, text_key):
        self.text_key = text_key

    def load_model(self, base_model_path, adapter_path):
        base_model = load_luar_as_sentence_transformer(base_model_path)
        base_model.max_seq_length = self.token_max_length
        base_model.tokenizer.pad_token = base_model.tokenizer.pad_token or base_model.tokenizer.eos_token

        self.model = PeftModel.from_pretrained(base_model, model_id=adapter_path)

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

        if torch.cuda.is_available():
            logging.info("Using CUDA")
            self.model.half().cuda()

    def encode(self, data):
        batch_size = 1
        logging.info("Setting batch size to 1 for author level embeddings with LUAR.")

        all_outputs = []

        for i in range(0, len(data), batch_size):
            raw_text = data[i:i+batch_size]
            output = self.model.encode(raw_text, show_progress_bar=False)
            all_outputs.extend([np.mean(output, axis=0).tolist()])

        return all_outputs