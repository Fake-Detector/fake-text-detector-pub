import torch
from transformers import pipeline


class Translator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-ru-en", device: torch.device = torch.device("cpu")):
        self.device = device
        self.languagePipe = pipeline("text-classification",
                                       model="papluca/xlm-roberta-base-language-detection",
                                       truncation=True)
        self.pipe = pipeline("translation", model=model_name, device=device)

    def translate(self, text: str) -> str:
        language = self.languagePipe(text)[0]['label']

        return text if language == 'en' else self.pipe(text)[0]['translation_text']
