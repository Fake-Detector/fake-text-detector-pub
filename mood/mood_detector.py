import torch
from transformers import pipeline


class MoodDetector:
    def __init__(self, model_name: str = "seara/rubert-tiny2-ru-go-emotions",
                 device: torch.device = torch.device("cpu")):
        self.device = device
        self.pipe = pipeline("text-classification", model=model_name, device=device)

    def get_mood(self, text: str) -> str:
        return self.pipe(text)[0]['label']
