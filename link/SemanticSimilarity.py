import torch
from sentence_transformers import SentenceTransformer, util


class SemanticSimilarity:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 device: torch.device = torch.device("cpu")):
        self.device = device
        self.model = SentenceTransformer(model_name)

    def similarity(self, original: str, compare: str) -> float:
        embedding_1 = self.model.encode(original, convert_to_tensor=True)
        embedding_2 = self.model.encode(compare, convert_to_tensor=True)

        return util.pytorch_cos_sim(embedding_1, embedding_2).item()
