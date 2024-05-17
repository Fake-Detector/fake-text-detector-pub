import logging
import re

import diff_match_patch as dmp_module
import nltk

from binoculars import Binoculars
from detector.fake_detector import FakeDetector
from grpc_api.protos import fake_detector_pb2_grpc, fake_detector_pb2, news_getter_pb2
from link.SourceComparing import SourceComparing
from mood.mood_detector import MoodDetector
from tags.tag_generator import TagGenerator
from transcribe.transcriber import Transcriber
from translator.translator import Translator


class FakeDetectorService(fake_detector_pb2_grpc.FakeDetectorServicer):
    def __init__(self, detector: FakeDetector, binoculars: Binoculars,
                 translator: Translator, tag_generator: TagGenerator,
                 mood_detector: MoodDetector, source_comparing: SourceComparing, transcriber: Transcriber):
        self.detector = detector
        self.binoculars = binoculars
        self.translator = translator
        self.tag_generator = tag_generator
        self.mood_detector = mood_detector
        self.source_comparing = source_comparing
        self.transcriber = transcriber

    def CheckTrust(self, request, context):
        logging.info(f"Get Request: {request}")

        predicted = self.detector.predict(request.text)

        return fake_detector_pb2.CheckTrustResponse(checking_result=predicted)

    def CheckAITrust(self, request, context):
        logging.info(f"Get Request: {request}")

        text = self.translator.translate(text=request.text)
        human_made_predicted = self.binoculars.predict(text).item()

        sentences = self.detector.formatter.split_text(request.text)

        sentences_predicted = [
            fake_detector_pb2.SentenceAiTrust(
                human_made=self.binoculars.predict(self.translator.translate(text=sentence)).item(), sentence=sentence)
            for sentence in sentences]

        return fake_detector_pb2.CheckAITrustResponse(overall_human_made=human_made_predicted,
                                                      sentences=sentences_predicted)

    def GenerateTags(self, request, context):
        logging.info(f"Get Request: {request}")

        text = self.translator.translate(text=request.text)
        tags = self.tag_generator.generate(text)

        return fake_detector_pb2.GenerateTagsResponse(tags=tags)

    def CheckMood(self, request, context):
        logging.info(f"Get Request: {request}")

        mood = self.mood_detector.get_mood(request.text)

        return fake_detector_pb2.CheckMoodResponse(mood=mood)

    def _get_query(self, text: str, count: int) -> str:
        return " ".join(re.findall(r'\b\w+\b', text)[:count])

    def CheckSources(self, request, context):
        logging.info(f"Get Request: {request}")

        source_comparing = self.source_comparing.get_sources_comparing(request.text)

        return fake_detector_pb2.CheckSourcesResponse(result=source_comparing)

    def GetTextImage(self, request, context):
        logging.info(f"Get Request: {request}")

        text = self.transcriber.transcribe_image(request.url)

        return fake_detector_pb2.GetTextImageResponse(text=text)

    def GetTextAudio(self, request, context):
        logging.info(f"Get Request: {request}")

        text = self.transcriber.transcribe_audio(request.url)

        return fake_detector_pb2.GetTextAudioResponse(text=text)
