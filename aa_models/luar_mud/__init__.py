import numpy as np
import os
from transformers import AutoModel, AutoTokenizer


class LUAR_MUD:
    def __init__(self) -> None:
        base_model_path = os.environ.get("LORA_BASEMODEL_CHECKPOINT_PATH", "rrivera1849/LUAR-MUD")
        #print('base model', base_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )

    def encode(self, sentences, batch_size=8, convert_to_numpy=True, max_length=512):
        """
        Wrapper
        """
        batched_outputs = []

        for i in range(0, len(sentences), batch_size):
            texts_chunk = sentences[i : i + batch_size]

            # We have one document per author
            episode_length = 1
            tokenized_text = self.tokenizer(
                texts_chunk,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # inputs size: (batch_size, episode_length, max_token_length)
            tokenized_text["input_ids"] = tokenized_text["input_ids"].reshape(
                -1, episode_length, max_length
            )
            tokenized_text["attention_mask"] = tokenized_text["attention_mask"].reshape(
                -1, episode_length, max_length
            )

            out = self.model(**tokenized_text)
            batched_outputs.append(
                out.squeeze().detach().numpy()
                if convert_to_numpy
                else out.squeeze().detach()
            )

        return np.vstack(batched_outputs)
