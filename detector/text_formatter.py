import string
import re


class TextFormatter:
    def __init__(self, stop_words, stemmer_ru, stemmer_en, punkt):
        self.stop_words = stop_words
        self.stemmer_ru = stemmer_ru
        self.stemmer_en = stemmer_en
        self.punkt = punkt

    @staticmethod
    def _remove_emojis(data: str):
        emojis = re.compile("["
                            u"\U0001F600-\U0001F64F"
                            u"\U0001F300-\U0001F5FF"
                            u"\U0001F680-\U0001F6FF"
                            u"\U0001F1E0-\U0001F1FF"
                            u"\U00002500-\U00002BEF"
                            u"\U00002702-\U000027B0"
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            u"\U0001f926-\U0001f937"
                            u"\U00010000-\U0010ffff"
                            u"\u2640-\u2642"
                            u"\u2600-\u2B55"
                            u"\u200d"
                            u"\u23cf"
                            u"\u23e9"
                            u"\u231a"
                            u"\ufe0f"
                            u"\u3030"
                            "]+", re.UNICODE)
        return re.sub(emojis, '', data)

    def preprocess_text(self, text: str) -> str:
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        text = text.replace('[^\w\s]', '')
        text = text.replace('\d+', '')
        text = TextFormatter._remove_emojis(text)
        text = [self.stemmer_ru.stem(word) for word in text.split() if word not in self.stop_words]
        text = " ".join(text)
        return text

    def split_text(self, text: str) -> list[str]:
        sentences: list[str] = self.punkt.tokenize(text)
        return [sentence.strip() for sentence in sentences if sentence.strip() != ""]
