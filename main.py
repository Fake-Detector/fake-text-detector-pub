import logging
import os

import grpc
import torch
from grpc_reflection.v1alpha import reflection
from concurrent import futures
import nltk
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from pydub import AudioSegment

from binoculars import Binoculars
from detector.fake_detector import FakeDetector
from detector.text_formatter import TextFormatter
from grpc_api.protos import fake_detector_pb2_grpc, fake_detector_pb2, news_getter_pb2_grpc
from grpc_api.services.fake_detector_service import FakeDetectorService
from link.KeywordsExtractor import KeywordsExtractor
from link.SourceComparing import SourceComparing
from link.TextComparing import TextComparing
from mood.mood_detector import MoodDetector
from link.SemanticSimilarity import SemanticSimilarity
from tags.tag_generator import TagGenerator
from transcribe.transcriber import Transcriber
from translator.translator import Translator


def service_init():
    logging.info(f"Start service init")
    logging.info(torch.cuda.is_available())
    if os.name == 'nt':
        AudioSegment.converter = r'./ffmpeg.exe'
        AudioSegment.ffprobe = r"./ffprobe.exe"
    nltk.download('stopwords')
    stop_words = stopwords.words('russian') + stopwords.words('english')
    stemmer_ru = SnowballStemmer("russian")
    stemmer_en = SnowballStemmer("english")
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    text_formatter = TextFormatter(stop_words, stemmer_ru, stemmer_en, tokenizer)
    device = torch.device("cpu")
    model_name = "TVI/fake-text-detector-model"
    binoculars = Binoculars(
        observer_name_or_path="google/gemma-2b",
        performer_name_or_path="google/gemma-2b-it"
    )
    translator = Translator()
    tag_generator = TagGenerator()
    mood_detector = MoodDetector()
    semantic_similarity = SemanticSimilarity()
    keywords_extractor = KeywordsExtractor()
    text_comparing = TextComparing(semantic_similarity, keywords_extractor)
    AudioSegment.converter = './ffmpeg.exe'
    AudioSegment.ffprobe = "./ffprobe.exe"
    transcriber = Transcriber(device=device)
    news_getter_channel = grpc.insecure_channel('localhost:50052')
    news_getter_stub = news_getter_pb2_grpc.NewsGetterStub(news_getter_channel)
    source_comparing = SourceComparing(news_getter_stub, text_comparing)

    fake_detector = FakeDetector(formatter=text_formatter, model_name=model_name, device=device)
    logging.info(f"End service init")

    return FakeDetectorService(detector=fake_detector, binoculars=binoculars,
                               translator=translator, tag_generator=tag_generator,
                               mood_detector=mood_detector, source_comparing=source_comparing, transcriber=transcriber)


def main():
    logging.basicConfig(level=logging.DEBUG)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service = service_init()
    fake_detector_pb2_grpc.add_FakeDetectorServicer_to_server(service, server)

    service_names = (
        fake_detector_pb2.DESCRIPTOR.services_by_name['FakeDetector'].full_name,
        reflection.SERVICE_NAME
    )

    reflection.enable_server_reflection(service_names, server)
    port = 50051
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logging.info(f"Server started on {port} port")
    server.wait_for_termination()


if __name__ == '__main__':
    main()
