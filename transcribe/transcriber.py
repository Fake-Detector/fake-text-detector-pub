import os
import tempfile

import easyocr
import requests
import torch
import whisper


class Transcriber:
    def __init__(self, model_name: str = "small", device: torch.device = torch.device("cpu")):
        self.reader = easyocr.Reader(['ru', 'en'], gpu=device != torch.device("cpu"))
        self.model = whisper.load_model(model_name)

    def transcribe_image(self, image_url: str) -> str:
        tokens = self.reader.readtext(image_url)
        return " ".join([token[1] for token in tokens])

    def transcribe_audio(self, media_url: str) -> str:
        response = requests.get(media_url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        result = self.model.transcribe(tmp_file_path, verbose=True)

        os.remove(tmp_file_path)

        return result["text"]
