from link.KeywordsExtractor import KeywordsExtractor
from link.SemanticSimilarity import SemanticSimilarity
import diff_match_patch as dmp_module


class TextComparing:
    def __init__(self, semantic_similarity: SemanticSimilarity, keywords_extractor: KeywordsExtractor):
        self.semantic_similarity = semantic_similarity
        self.keywords_extractor = keywords_extractor

    def diff_comparing(self, original: str, compare: str) -> list[tuple[int, str]]:
        difference_module = dmp_module.diff_match_patch()
        difference = difference_module.diff_main(original, compare)
        difference_module.diff_cleanupSemantic(difference)
        return difference

    def semantic_comparing(self, original: str, compare: str) -> float:
        return self.semantic_similarity.similarity(original, compare)

    def _intersect_keywords(self, original: list[str], compare: list[str]) -> list[str]:
        intersect_keywords = set()
        for keyword_original in original:
            for keyword_compare in compare:
                if self.semantic_similarity.similarity(keyword_original, keyword_compare) > 0.85:
                    intersect_keywords.add(keyword_original)

        return list(intersect_keywords)

    def keywords_comparing(self, original: str, compare: str) -> tuple[
        dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
        original_keywords = self.keywords_extractor.extract_keywords(original)
        compare_keywords = self.keywords_extractor.extract_keywords(compare)
        result = {}
        all_keys = set(original_keywords.keys()).union(set(compare_keywords.keys()))

        for key in all_keys:
            if key in original_keywords and key in compare_keywords:
                result[key] = self._intersect_keywords(original_keywords[key], compare_keywords[key])

        return original_keywords, compare_keywords, result
