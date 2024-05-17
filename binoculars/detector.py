from typing import Union, Tuple, Any

import math
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import huggingface_config
from .utils import assert_tokenizer_consistency
from .metrics import perplexity, entropy

torch.set_grad_enabled(False)

BINOCULARS_ACCURACY_THRESHOLD = 0.934
BINOCULARS_FPR_THRESHOLD = 0.875

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = DEVICE_1


class Binoculars(object):
    def __init__(self,
                 observer_name_or_path: str = "tiiuae/falcon-7b",
                 performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 mode: str = "low-fpr",
                 ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.change_mode(mode)
        self.observer_model = AutoModelForCausalLM.from_pretrained(observer_name_or_path,
                                                                   trust_remote_code=True,
                                                                   torch_dtype=torch.bfloat16 if use_bfloat16
                                                                   else torch.float32,
                                                                   token=huggingface_config["TOKEN"],
                                                                   cache_dir='./huggingface/cache/'
                                                                   ).to(DEVICE_2)
        self.performer_model = AutoModelForCausalLM.from_pretrained(performer_name_or_path,
                                                                    trust_remote_code=True,
                                                                    torch_dtype=torch.bfloat16 if use_bfloat16
                                                                    else torch.float32,
                                                                    token=huggingface_config["TOKEN"],
                                                                    cache_dir='./huggingface/cache/'
                                                                    ).to(DEVICE_2)

        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path, token=huggingface_config["TOKEN"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> tuple[torch.Tensor, torch.Tensor]:
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores

    def sigmoid(self, x) -> float:
        return 1 / (1 + math.exp(-25 * x))

    def predict(self, input_text: Union[list[str], str]) -> Union[list[float], float]:
        binoculars_scores = np.array(self.compute_score(input_text))
        sigmoid_f = np.vectorize(lambda x: self.sigmoid(x - self.threshold) * 100)
        pred = sigmoid_f(binoculars_scores)
        return pred
