from span_marker import SpanMarkerModel


class KeywordsExtractor:
    def __init__(self, model_name: str = "tomaarsen/span-marker-mbert-base-multinerd"):
        self.model = SpanMarkerModel.from_pretrained(model_name)

    def extract_keywords(self, text) -> dict[str, list[str]]:
        keywords = self.model.predict(text)
        result = {}
        for keyword in keywords:
            label: str = keyword['label']
            value: str = keyword['span']
            if label in result:
                result[label].append(value)
            else:
                result[label] = [value]

        return result
