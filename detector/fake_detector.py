import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from detector.text_formatter import TextFormatter


class FakeDetector:
    def __init__(self, formatter: TextFormatter, model_name: str, device: torch.device):
        self.formatter = formatter
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(
            device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, text: str):
        tokenized = self._tokenize(text)
        with torch.no_grad():
            outputs = self.model(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask'])
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_prob = probabilities[:, 1].item() * 100

        return predicted_prob

    def _tokenize(self, text: str) -> dict:
        text = self.formatter.preprocess_text(text)

        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

        tokenized = {'input_ids': torch.tensor([encodings['input_ids'].squeeze().numpy().tolist()]).to(self.device),
                     'attention_mask': torch.tensor([encodings['attention_mask'].squeeze().numpy().tolist()]).to(
                         self.device)}

        return tokenized
