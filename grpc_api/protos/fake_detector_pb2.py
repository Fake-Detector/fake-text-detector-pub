# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: fake_detector.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import wrappers_pb2 as google_dot_protobuf_dot_wrappers__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x13\x66\x61ke_detector.proto\x12\rfake_detector\x1a\x1egoogle/protobuf/wrappers.proto\"!\n\x11\x43heckTrustRequest\x12\x0c\n\x04text\x18\x01 \x01(\t\"-\n\x12\x43heckTrustResponse\x12\x17\n\x0f\x63hecking_result\x18\x01 \x01(\x01\"#\n\x13\x43heckAITrustRequest\x12\x0c\n\x04text\x18\x01 \x01(\t\"e\n\x14\x43heckAITrustResponse\x12\x1a\n\x12overall_human_made\x18\x01 \x01(\x01\x12\x31\n\tsentences\x18\x02 \x03(\x0b\x32\x1e.fake_detector.SentenceAiTrust\"7\n\x0fSentenceAiTrust\x12\x12\n\nhuman_made\x18\x01 \x01(\x01\x12\x10\n\x08sentence\x18\x02 \x01(\t\"#\n\x13GenerateTagsRequest\x12\x0c\n\x04text\x18\x01 \x01(\t\"$\n\x14GenerateTagsResponse\x12\x0c\n\x04tags\x18\x01 \x03(\t\" \n\x10\x43heckMoodRequest\x12\x0c\n\x04text\x18\x01 \x01(\t\"!\n\x11\x43heckMoodResponse\x12\x0c\n\x04mood\x18\x01 \x01(\t\"#\n\x13\x43heckSourcesRequest\x12\x0c\n\x04text\x18\x01 \x01(\t\"C\n\x14\x43heckSourcesResponse\x12+\n\x06result\x18\x01 \x03(\x0b\x32\x1b.fake_detector.SourceResult\"\xb7\x02\n\x0cSourceResult\x12\x0b\n\x03url\x18\x01 \x01(\t\x12\x39\n\x13semantic_similarity\x18\x02 \x01(\x0b\x32\x1c.google.protobuf.DoubleValue\x12\x36\n\x0ftext_comparison\x18\x03 \x03(\x0b\x32\x1d.fake_detector.DiffComparison\x12<\n\x12keyword_comparison\x18\x04 \x01(\x0b\x32 .fake_detector.KeywordsComparing\x12\x33\n\roriginal_text\x18\x05 \x01(\x0b\x32\x1c.google.protobuf.StringValue\x12\x34\n\x0eoriginal_title\x18\x06 \x01(\x0b\x32\x1c.google.protobuf.StringValue\"5\n\x0e\x44iffComparison\x12\x14\n\x0c\x63ompare_type\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\t\"\x94\x01\n\x11KeywordsComparing\x12(\n\x08original\x18\x01 \x03(\x0b\x32\x16.fake_detector.Keyword\x12\'\n\x07\x63ompare\x18\x02 \x03(\x0b\x32\x16.fake_detector.Keyword\x12,\n\x0cintersection\x18\x03 \x03(\x0b\x32\x16.fake_detector.Keyword\"&\n\x07Keyword\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x0e\n\x06values\x18\x02 \x03(\t\"\"\n\x13GetTextImageRequest\x12\x0b\n\x03url\x18\x01 \x01(\t\"$\n\x14GetTextImageResponse\x12\x0c\n\x04text\x18\x01 \x01(\t\"\"\n\x13GetTextAudioRequest\x12\x0b\n\x03url\x18\x01 \x01(\t\"$\n\x14GetTextAudioResponse\x12\x0c\n\x04text\x18\x01 \x01(\t2\xee\x04\n\x0c\x46\x61keDetector\x12Q\n\nCheckTrust\x12 .fake_detector.CheckTrustRequest\x1a!.fake_detector.CheckTrustResponse\x12W\n\x0c\x43heckAITrust\x12\".fake_detector.CheckAITrustRequest\x1a#.fake_detector.CheckAITrustResponse\x12W\n\x0cGenerateTags\x12\".fake_detector.GenerateTagsRequest\x1a#.fake_detector.GenerateTagsResponse\x12N\n\tCheckMood\x12\x1f.fake_detector.CheckMoodRequest\x1a .fake_detector.CheckMoodResponse\x12W\n\x0c\x43heckSources\x12\".fake_detector.CheckSourcesRequest\x1a#.fake_detector.CheckSourcesResponse\x12W\n\x0cGetTextImage\x12\".fake_detector.GetTextImageRequest\x1a#.fake_detector.GetTextImageResponse\x12W\n\x0cGetTextAudio\x12\".fake_detector.GetTextAudioRequest\x1a#.fake_detector.GetTextAudioResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'fake_detector_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _globals['_CHECKTRUSTREQUEST']._serialized_start=70
  _globals['_CHECKTRUSTREQUEST']._serialized_end=103
  _globals['_CHECKTRUSTRESPONSE']._serialized_start=105
  _globals['_CHECKTRUSTRESPONSE']._serialized_end=150
  _globals['_CHECKAITRUSTREQUEST']._serialized_start=152
  _globals['_CHECKAITRUSTREQUEST']._serialized_end=187
  _globals['_CHECKAITRUSTRESPONSE']._serialized_start=189
  _globals['_CHECKAITRUSTRESPONSE']._serialized_end=290
  _globals['_SENTENCEAITRUST']._serialized_start=292
  _globals['_SENTENCEAITRUST']._serialized_end=347
  _globals['_GENERATETAGSREQUEST']._serialized_start=349
  _globals['_GENERATETAGSREQUEST']._serialized_end=384
  _globals['_GENERATETAGSRESPONSE']._serialized_start=386
  _globals['_GENERATETAGSRESPONSE']._serialized_end=422
  _globals['_CHECKMOODREQUEST']._serialized_start=424
  _globals['_CHECKMOODREQUEST']._serialized_end=456
  _globals['_CHECKMOODRESPONSE']._serialized_start=458
  _globals['_CHECKMOODRESPONSE']._serialized_end=491
  _globals['_CHECKSOURCESREQUEST']._serialized_start=493
  _globals['_CHECKSOURCESREQUEST']._serialized_end=528
  _globals['_CHECKSOURCESRESPONSE']._serialized_start=530
  _globals['_CHECKSOURCESRESPONSE']._serialized_end=597
  _globals['_SOURCERESULT']._serialized_start=600
  _globals['_SOURCERESULT']._serialized_end=911
  _globals['_DIFFCOMPARISON']._serialized_start=913
  _globals['_DIFFCOMPARISON']._serialized_end=966
  _globals['_KEYWORDSCOMPARING']._serialized_start=969
  _globals['_KEYWORDSCOMPARING']._serialized_end=1117
  _globals['_KEYWORD']._serialized_start=1119
  _globals['_KEYWORD']._serialized_end=1157
  _globals['_GETTEXTIMAGEREQUEST']._serialized_start=1159
  _globals['_GETTEXTIMAGEREQUEST']._serialized_end=1193
  _globals['_GETTEXTIMAGERESPONSE']._serialized_start=1195
  _globals['_GETTEXTIMAGERESPONSE']._serialized_end=1231
  _globals['_GETTEXTAUDIOREQUEST']._serialized_start=1233
  _globals['_GETTEXTAUDIOREQUEST']._serialized_end=1267
  _globals['_GETTEXTAUDIORESPONSE']._serialized_start=1269
  _globals['_GETTEXTAUDIORESPONSE']._serialized_end=1305
  _globals['_FAKEDETECTOR']._serialized_start=1308
  _globals['_FAKEDETECTOR']._serialized_end=1930
# @@protoc_insertion_point(module_scope)
