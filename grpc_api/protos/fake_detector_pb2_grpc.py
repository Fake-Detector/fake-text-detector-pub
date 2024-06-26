# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from grpc_api.protos import fake_detector_pb2 as fake__detector__pb2


class FakeDetectorStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.CheckTrust = channel.unary_unary(
                '/fake_detector.FakeDetector/CheckTrust',
                request_serializer=fake__detector__pb2.CheckTrustRequest.SerializeToString,
                response_deserializer=fake__detector__pb2.CheckTrustResponse.FromString,
                )
        self.CheckAITrust = channel.unary_unary(
                '/fake_detector.FakeDetector/CheckAITrust',
                request_serializer=fake__detector__pb2.CheckAITrustRequest.SerializeToString,
                response_deserializer=fake__detector__pb2.CheckAITrustResponse.FromString,
                )
        self.GenerateTags = channel.unary_unary(
                '/fake_detector.FakeDetector/GenerateTags',
                request_serializer=fake__detector__pb2.GenerateTagsRequest.SerializeToString,
                response_deserializer=fake__detector__pb2.GenerateTagsResponse.FromString,
                )
        self.CheckMood = channel.unary_unary(
                '/fake_detector.FakeDetector/CheckMood',
                request_serializer=fake__detector__pb2.CheckMoodRequest.SerializeToString,
                response_deserializer=fake__detector__pb2.CheckMoodResponse.FromString,
                )
        self.CheckSources = channel.unary_unary(
                '/fake_detector.FakeDetector/CheckSources',
                request_serializer=fake__detector__pb2.CheckSourcesRequest.SerializeToString,
                response_deserializer=fake__detector__pb2.CheckSourcesResponse.FromString,
                )
        self.GetTextImage = channel.unary_unary(
                '/fake_detector.FakeDetector/GetTextImage',
                request_serializer=fake__detector__pb2.GetTextImageRequest.SerializeToString,
                response_deserializer=fake__detector__pb2.GetTextImageResponse.FromString,
                )
        self.GetTextAudio = channel.unary_unary(
                '/fake_detector.FakeDetector/GetTextAudio',
                request_serializer=fake__detector__pb2.GetTextAudioRequest.SerializeToString,
                response_deserializer=fake__detector__pb2.GetTextAudioResponse.FromString,
                )


class FakeDetectorServicer(object):
    """Missing associated documentation comment in .proto file."""

    def CheckTrust(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CheckAITrust(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GenerateTags(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CheckMood(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CheckSources(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetTextImage(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetTextAudio(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FakeDetectorServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'CheckTrust': grpc.unary_unary_rpc_method_handler(
                    servicer.CheckTrust,
                    request_deserializer=fake__detector__pb2.CheckTrustRequest.FromString,
                    response_serializer=fake__detector__pb2.CheckTrustResponse.SerializeToString,
            ),
            'CheckAITrust': grpc.unary_unary_rpc_method_handler(
                    servicer.CheckAITrust,
                    request_deserializer=fake__detector__pb2.CheckAITrustRequest.FromString,
                    response_serializer=fake__detector__pb2.CheckAITrustResponse.SerializeToString,
            ),
            'GenerateTags': grpc.unary_unary_rpc_method_handler(
                    servicer.GenerateTags,
                    request_deserializer=fake__detector__pb2.GenerateTagsRequest.FromString,
                    response_serializer=fake__detector__pb2.GenerateTagsResponse.SerializeToString,
            ),
            'CheckMood': grpc.unary_unary_rpc_method_handler(
                    servicer.CheckMood,
                    request_deserializer=fake__detector__pb2.CheckMoodRequest.FromString,
                    response_serializer=fake__detector__pb2.CheckMoodResponse.SerializeToString,
            ),
            'CheckSources': grpc.unary_unary_rpc_method_handler(
                    servicer.CheckSources,
                    request_deserializer=fake__detector__pb2.CheckSourcesRequest.FromString,
                    response_serializer=fake__detector__pb2.CheckSourcesResponse.SerializeToString,
            ),
            'GetTextImage': grpc.unary_unary_rpc_method_handler(
                    servicer.GetTextImage,
                    request_deserializer=fake__detector__pb2.GetTextImageRequest.FromString,
                    response_serializer=fake__detector__pb2.GetTextImageResponse.SerializeToString,
            ),
            'GetTextAudio': grpc.unary_unary_rpc_method_handler(
                    servicer.GetTextAudio,
                    request_deserializer=fake__detector__pb2.GetTextAudioRequest.FromString,
                    response_serializer=fake__detector__pb2.GetTextAudioResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'fake_detector.FakeDetector', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class FakeDetector(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def CheckTrust(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/fake_detector.FakeDetector/CheckTrust',
            fake__detector__pb2.CheckTrustRequest.SerializeToString,
            fake__detector__pb2.CheckTrustResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CheckAITrust(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/fake_detector.FakeDetector/CheckAITrust',
            fake__detector__pb2.CheckAITrustRequest.SerializeToString,
            fake__detector__pb2.CheckAITrustResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GenerateTags(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/fake_detector.FakeDetector/GenerateTags',
            fake__detector__pb2.GenerateTagsRequest.SerializeToString,
            fake__detector__pb2.GenerateTagsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CheckMood(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/fake_detector.FakeDetector/CheckMood',
            fake__detector__pb2.CheckMoodRequest.SerializeToString,
            fake__detector__pb2.CheckMoodResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CheckSources(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/fake_detector.FakeDetector/CheckSources',
            fake__detector__pb2.CheckSourcesRequest.SerializeToString,
            fake__detector__pb2.CheckSourcesResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetTextImage(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/fake_detector.FakeDetector/GetTextImage',
            fake__detector__pb2.GetTextImageRequest.SerializeToString,
            fake__detector__pb2.GetTextImageResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetTextAudio(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/fake_detector.FakeDetector/GetTextAudio',
            fake__detector__pb2.GetTextAudioRequest.SerializeToString,
            fake__detector__pb2.GetTextAudioResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
