import os
import shutil
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer

def fix_tokenizer(tokenizer):
    # Make the FastTokenizer savable
    def _override_save_pretrained(save_directory, **kwargs):
        tokenizer_files = [
            os.path.join(tokenizer.name_or_path, f) for f in os.listdir(tokenizer.name_or_path)
            if 'token' in f or 'vocab' in f
        ]
        for t_file in tokenizer_files:
            shutil.copyfile(t_file, os.path.join(save_directory, os.path.basename(t_file)))
    tokenizer.save_pretrained = _override_save_pretrained
    return tokenizer

def load_luar_as_sentence_transformer(model_name_or_path, **kwargs):
    t_module = Transformer(model_name_or_path,
        model_args={"trust_remote_code": True},
        config_args={"trust_remote_code": True}
    )
    st = SentenceTransformer(
        None,
        modules=[t_module],
        **kwargs,
    )
    t_module.tokenizer = fix_tokenizer(t_module.tokenizer)

    # Make Sentence Transformers compatible with LUAR
    # Makes Transformer() return sentence_embeddings directy, no pooling layer needed
    # since LUAR returns embeddings directly
    def wrap_old_transformer_forward(features):
        trans_features = {
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"]
        }
        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]
        return st._modules['0'].auto_model(**trans_features, return_dict=False)

    st._modules['0'].forward = wrap_old_transformer_forward

    # Make LUAR compatible with Sentence Transformers
    # (makes the input_ids and attention_mask the correct shape, episode_length=1 or 1 text per author)
    old_forward = st._modules['0'].auto_model.forward
    def wrap_old_forward(*args, **kwargs):
        kwargs['input_ids'] = kwargs['input_ids'].unsqueeze(1)
        kwargs['attention_mask'] = kwargs['attention_mask'].unsqueeze(1)
        return {'sentence_embedding': old_forward(*args, **kwargs)}
    st._modules['0'].auto_model.forward = lambda *args, return_dict, **kwargs: wrap_old_forward(
        *args,
        **kwargs,
    )

    # Return ST model
    return st

def get_luar_trainer():
    # LUAR always uses a FastTokenizer which is not compatible with multi-processing
    # See: https://stackoverflow.com/a/67254879
    os.sched_getaffinity = lambda *args: [None]

    import torch
    from typing import Any
    from functools import partial, cached_property
    from datadreamer.trainers import TrainSentenceTransformer
    from datadreamer.embedders.sentence_transformers_embedder import _normalize_model_name
    from datadreamer.utils.background_utils import RunIfTimeout
    from datadreamer.utils.import_utils import ignore_transformers_warnings
    from datadreamer.utils.arg_utils import DEFAULT, Default, default_to
    from datadreamer.utils.hf_model_utils import (
        get_tokenizer,
        get_model_max_context_length,
        validate_peft_config
    )

    class TrainLUAR(TrainSentenceTransformer):
        @cached_property
        def tokenizer(self):
            tokenizer = fix_tokenizer(super().tokenizer)
            return tokenizer

        def _create_model(
            self,
            label2id: None | dict[int, Any] = None,
            id2label: None | dict[int, Any] = None,
            is_multi_target: bool = False,
            device: None
            | int
            | str
            | torch.device
            | list[int | str | torch.device]
            | Default = DEFAULT,
            is_ref_model: bool = False,
        ) -> SentenceTransformer:
            # Seed
            if self.seed:
                torch.manual_seed(self.seed)
                if torch.cuda.is_available():  # pragma: no cover
                    torch.cuda.manual_seed_all(self.seed)

            # Load model
            log_if_timeout = RunIfTimeout(
                partial(lambda self: self.logger.info("Loading model..."), self),
                timeout=10.0,
            )
            model_device = default_to(device, self.device)
            model = load_luar_as_sentence_transformer(
                self.model_name,
                device="cpu" if isinstance(model_device, list) else model_device,
                **self.kwargs
            )
            model[0].tokenizer = fix_tokenizer(get_tokenizer(
                _normalize_model_name(self.model_name),
                revision=None,
                trust_remote_code=True,
            ))
            model.max_seq_length = (
                get_model_max_context_length(
                    model_name=self.model_name, config=model[0].auto_model.config
                )
                if model.max_seq_length is None
                else model.max_seq_length
            )
            self.max_seq_length = model.max_seq_length

            # Set model dtype
            model = model.to(self.dtype)

            # Create PeftModel if peft_config
            if self.peft_config:
                # Two warnings we can't silence are thrown by peft at import-time so
                # we import this library only when needed
                with ignore_transformers_warnings():
                    from peft import get_peft_model, prepare_model_for_kbit_training

                if self.quantization_config:  # pragma: no cover
                    model = prepare_model_for_kbit_training(model)
                model = get_peft_model(model, validate_peft_config(model, self.peft_config))

            # Switch model to train mode
            model.train()

            # Finished loading
            log_if_timeout.stop(
                partial(lambda self: self.logger.info("Finished loading."), self)
            )

            return model

        def _load_model(
            self,
            label2id: None | dict[Any, int] = None,
            id2label: None | dict[int, Any] = None,
            is_multi_target: bool = False,
            with_optimizations: bool = True,
        ) -> SentenceTransformer:
            # Load model metadata
            self._load_model_metadata()

            # Load model
            log_if_timeout = RunIfTimeout(
                partial(
                    lambda self: self.logger.info("Loading trained model from disk..."),
                    self,
                ),
                timeout=10.0,
            )
            if self.peft_config:
                # Two warnings we can't silence are thrown by peft at import-time so
                # we import this library only when needed
                with ignore_transformers_warnings():
                    from peft import PeftModel

                model = model = load_luar_as_sentence_transformer(
                    self.model_name,
                    device="cpu" if isinstance(self.device, list) else self.device,
                    **self.kwargs
                )
                model[0].tokenizer = fix_tokenizer(get_tokenizer(
                    _normalize_model_name(self.model_name),
                    revision=None,
                    trust_remote_code=True,
                ))
                model.max_seq_length = (
                    get_model_max_context_length(
                        model_name=self.model_name, config=model[0].auto_model.config
                    )
                    if model.max_seq_length is None
                    else model.max_seq_length
                )
                model = PeftModel.from_pretrained(
                    model,
                    model_id=os.path.join(self._output_folder_path, "_model"),
                    torch_dtype=self.dtype,
                    **self.kwargs,
                )
            else:
                model = load_luar_as_sentence_transformer(
                    self.model_name,
                    device="cpu" if isinstance(self.device, list) else self.device,
                    **self.kwargs,
                )
            self.max_seq_length = model.max_seq_length

            # Set model dtype
            model = model.to(self.dtype)

            # Switch model to eval mode
            model.eval()

            if with_optimizations:
                # Torch compile
                # torch._dynamo.config.suppress_errors = True
                # model = torch.compile(model)
                pass

            # Finished loading
            log_if_timeout.stop(
                partial(lambda self: self.logger.info("Finished loading."), self)
            )

            return model

    return TrainLUAR