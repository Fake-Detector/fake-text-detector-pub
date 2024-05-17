import re

from grpc_api.protos import news_getter_pb2, fake_detector_pb2
from grpc_api.protos.news_getter_pb2_grpc import NewsGetterStub
from link.TextComparing import TextComparing


class SourceComparing:
    def __init__(self, news_getter_client: NewsGetterStub, text_comparing: TextComparing):
        self.news_getter_client = news_getter_client
        self.text_comparing = text_comparing

    @staticmethod
    def _get_query(text: str, count: int) -> str:
        return " ".join(re.findall(r'\b\w+\b', text)[:count])

    def _get_links(self, text) -> set[str]:
        query = SourceComparing._get_query(text, 10)

        links_with_scrapers = self.news_getter_client.SearchNews(
            news_getter_pb2.SearchNewsRequest(query=query,
                                              max_site_links=1,
                                              sites=[
                                                  news_getter_pb2.Site.LentaRu,
                                                  news_getter_pb2.Site.IzRu,
                                                  news_getter_pb2.Site.Ria,
                                                  news_getter_pb2.Site.Interfax,
                                                  news_getter_pb2.Site.Tass,
                                                  news_getter_pb2.Site.BBC,
                                                  news_getter_pb2.Site.CNN
                                              ])).links

        common_links = self.news_getter_client.SearchNews(
            news_getter_pb2.SearchNewsRequest(query=query, max_site_links=10)).links

        return set(list(common_links) + list(links_with_scrapers))

    @staticmethod
    def _prepare_keywords(keywords: dict[str, list[str]]):
        return [fake_detector_pb2.Keyword(key=key, values=keywords[key]) for key in keywords.keys()]

    def get_sources_comparing(self, text: str) -> list[fake_detector_pb2.SourceResult]:
        result_links = self._get_links(text)
        result_infos = []

        for link in result_links:
            try:
                result = self.news_getter_client.GetNewsContent(
                    news_getter_pb2.GetNewsContentRequest(url=link, scraper=news_getter_pb2.Site.AutoDetect))
                if result.is_success:
                    similarity = self.text_comparing.semantic_comparing(text, result.content)
                    if similarity < 0.5:
                        continue

                    difference = self.text_comparing.diff_comparing(text, result.content)
                    text_comparison = [
                        fake_detector_pb2.DiffComparison(compare_type=item[0], value=item[1]) for item in difference]

                    original_keyword, compare_keyword, result_keyword = self.text_comparing.keywords_comparing(
                        text,
                        result.content)

                    keyword_comparison = fake_detector_pb2.KeywordsComparing(
                        original=SourceComparing._prepare_keywords(original_keyword),
                        compare=SourceComparing._prepare_keywords(compare_keyword),
                        intersection=SourceComparing._prepare_keywords(result_keyword))

                    source_result = fake_detector_pb2.SourceResult(url=result.url,
                                                                   text_comparison=text_comparison,
                                                                   keyword_comparison=keyword_comparison)
                    source_result.original_text.value = result.content
                    source_result.original_title.value = result.title
                    source_result.semantic_similarity.value = similarity

                    result_infos.append(source_result)
                else:
                    result_infos.append(fake_detector_pb2.SourceResult(url=link))
            except Exception as error:
                print(f"Error getting news: {error}")
                result_infos.append(fake_detector_pb2.SourceResult(url=link))

        return result_infos
