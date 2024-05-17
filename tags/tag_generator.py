import torch
from transformers import pipeline


class TagGenerator:
    def __init__(self, model_name: str = "fabiochiu/t5-base-tag-generation",
                 device: torch.device = torch.device("cpu")):
        self.device = device
        self.pipe = pipeline("text2text-generation", model=model_name)

    def generate(self, text: str) -> list[str]:
        return list(set([item.strip() for item in self.pipe(text)[0]['generated_text'].split(',') if item.strip()]))
