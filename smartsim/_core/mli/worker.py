# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import io
import pickle
import typing as t
from abc import ABC, abstractmethod

import torch

import smartsim.error as sse
from smartsim.log import get_logger

from .infrastructure import CommChannel, DragonCommChannel, FeatureStore
from .message_handler import MessageHandler

if t.TYPE_CHECKING:
    import dragon.channels as dch
    import dragon.utils as du


logger = get_logger(__name__)


class InferenceRequest:
    """Temporary model of an inference request"""

    def __init__(
        self,
        model_key: t.Optional[str] = None,
        callback: t.Optional[CommChannel] = None,
        raw_inputs: t.Optional[t.List[bytes]] = None,
        input_keys: t.Optional[t.List[str]] = None,
        output_keys: t.Optional[t.List[str]] = None,
        raw_model: t.Optional[bytes] = None,
        batch_size: int = 0,
        device: t.Optional[str] = None,
    ):
        """Initialize the InferenceRequest"""
        self.model_key = model_key
        self.raw_model = raw_model
        self.callback = callback
        self.raw_inputs = raw_inputs
        self.input_keys = input_keys or []
        self.output_keys = output_keys or []
        self.batch_size = batch_size
        self.device = device

    # @staticmethod
    # def from_dict(_source: t.Dict[str, t.Any]) -> "InferenceRequest":
    #     return InferenceRequest(
    #         # **source
    #     )


class InferenceReply:
    """Temporary model of a reply to an inference request"""

    def __init__(
        self,
        outputs: t.Optional[t.Collection[t.Any]] = None,
        output_keys: t.Optional[t.Collection[str]] = None,
        failed: bool = False,
    ) -> None:
        """Initialize the InferenceReply"""
        self.outputs: t.Collection[t.Any] = outputs or []
        self.output_keys: t.Collection[t.Optional[str]] = output_keys or []
        self.failed = failed


class ModelLoadResult:
    """A wrapper around a loaded model"""

    def __init__(self, model: t.Any) -> None:
        """Initialize the ModelLoadResult"""
        self.model = model


class InputTransformResult:
    """A wrapper around a transformed input"""

    def __init__(self, result: t.Any) -> None:
        """Initialize the InputTransformResult"""
        self.transformed = result


class ExecuteResult:
    """A wrapper around inference results"""

    def __init__(self, result: t.Any) -> None:
        """Initialize the ExecuteResult"""
        self.predictions = result


class InputFetchResult:
    """A wrapper around fetched inputs"""

    def __init__(self, result: t.Any) -> None:
        """Initialize the InputFetchResult"""
        self.inputs = result


class OutputTransformResult:
    """A wrapper around inference results transformed for transmission"""

    def __init__(self, result: t.Any) -> None:
        """Initialize the OutputTransformResult"""
        self.outputs = result


class BatchResult:
    """A wrapper around batched inputs"""

    def __init__(self, result: t.Any) -> None:
        """Initialize the BatchResult"""
        self.batch = result


class FetchModelResult:
    """A wrapper around raw fetched models"""

    def __init__(self, result: bytes) -> None:
        """Initialize the BatchResult"""
        self.model_bytes = result


class MachineLearningWorkerCore:
    """Basic functionality of ML worker that is shared across all worker types"""

    @staticmethod
    def fetch_model(
        request: InferenceRequest, feature_store: FeatureStore
    ) -> FetchModelResult:
        """Given a resource key, retrieve the raw model from a feature store
        :param request: The request that triggered the pipeline
        :param feature_store: The feature store used for persistence
        :return: Raw bytes of the model"""
        if request.raw_model:
            # Should we cache model in the feature store?
            # model_key = hash(request.raw_model)
            # feature_store[model_key] = request.raw_model
            # short-circuit and return the directly supplied model
            return FetchModelResult(request.raw_model)

        if not request.model_key:
            raise sse.SmartSimError(
                "Key must be provided to retrieve model from feature store"
            )

        try:
            raw_bytes = feature_store[request.model_key]
            return FetchModelResult(raw_bytes)
        except FileNotFoundError as ex:
            logger.exception(ex)
            raise sse.SmartSimError(
                f"Model could not be retrieved with key {request.model_key}"
            ) from ex

    @staticmethod
    def fetch_inputs(
        request: InferenceRequest, feature_store: FeatureStore
    ) -> InputFetchResult:
        """Given a collection of ResourceKeys, identify the physical location
        and input metadata
        :param request: The request that triggered the pipeline
        :param feature_store: The feature store used for persistence
        :return: the fetched input"""
        if request.input_keys:
            data: t.List[bytes] = []
            for input_ in request.input_keys:
                try:
                    tensor_bytes = feature_store[input_]
                    data.append(tensor_bytes)
                except KeyError as ex:
                    logger.exception(ex)
                    raise sse.SmartSimError(
                        f"Model could not be retrieved with key {input_}"
                    ) from ex
            return InputFetchResult(data)

        if request.raw_inputs:
            return InputFetchResult(request.raw_inputs)

        raise ValueError("No input source")

    @staticmethod
    def batch_requests(
        request: InferenceRequest, transform_result: InputTransformResult
    ) -> BatchResult:
        """Create a batch of requests. Return the batch when batch_size datum have been
        collected or a configured batch duration has elapsed.
        :param request: The request that triggered the pipeline
        :param transform_result: Transformed inputs ready for batching
        :return: `None` if batch size has not been reached and timeout not exceeded."""
        if transform_result is not None or request.batch_size:
            raise NotImplementedError("Batching is not yet supported")
        return BatchResult(None)

    @staticmethod
    def place_output(
        request: InferenceRequest,
        execute_result: ExecuteResult,
        feature_store: FeatureStore,
    ) -> t.Collection[t.Optional[str]]:
        """Given a collection of data, make it available as a shared resource in the
        feature store
        :param request: The request that triggered the pipeline
        :param execute_result: Results from inference
        :param feature_store: The feature store used for persistence"""
        keys: t.List[t.Optional[str]] = []
        # need to decide how to get back to original sub-batch inputs so they can be
        # accurately placed, datum might need to include this.

        for k, v in zip(request.output_keys, execute_result.predictions):
            feature_store[k] = v
            keys.append(k)

        return keys


class MachineLearningWorkerBase(MachineLearningWorkerCore, ABC):
    """Abstrct base class providing contract for a machine learning
    worker implementation."""

    @staticmethod
    @abstractmethod
    def deserialize(data_blob: bytes) -> InferenceRequest:
        """Given a collection of data serialized to bytes, convert the bytes
        to a proper representation used by the ML backend
        :param data_blob: inference request as a byte-serialized blob
        :return: InferenceRequest deserialized from the input"""

    @staticmethod
    @abstractmethod
    def load_model(
        request: InferenceRequest, fetch_result: FetchModelResult
    ) -> ModelLoadResult:
        """Given a loaded MachineLearningModel, ensure it is loaded into
        device memory
        :param request: The request that triggered the pipeline
        :return: ModelLoadResult wrapping the model loaded for the request"""

    @staticmethod
    @abstractmethod
    def transform_input(
        request: InferenceRequest, fetch_result: InputFetchResult
    ) -> InputTransformResult:
        """Given a collection of data, perform a transformation on the data
        :param request: The request that triggered the pipeline
        :param fetch_result: Raw output from fetching inputs out of a feature store
        :return: The transformed inputs wrapped in a InputTransformResult"""

    @staticmethod
    @abstractmethod
    def execute(
        request: InferenceRequest,
        load_result: ModelLoadResult,
        transform_result: InputTransformResult,
    ) -> ExecuteResult:
        """Execute an ML model on inputs transformed for use by the model
        :param request: The request that triggered the pipeline
        :param load_result: The result of loading the model onto device memory
        :param transform_result: The result of transforming inputs for model consumption
        :return: The result of inference wrapped in an ExecuteResult"""

    @staticmethod
    @abstractmethod
    def transform_output(
        request: InferenceRequest,
        execute_result: ExecuteResult,
    ) -> OutputTransformResult:
        """Given inference results, perform transformations required to
        transmit results to the requestor.
        :param request: The request that triggered the pipeline
        :param execute_result: The result of inference wrapped in an ExecuteResult
        :return:"""

    @staticmethod
    @abstractmethod
    def serialize_reply(reply: InferenceReply) -> t.Any:
        """Given an output, serialize to bytes for transport
        :param reply: The result of the inference pipeline
        :return: a byte-serialized version of the reply"""


class SampleTorchWorker(MachineLearningWorkerBase):
    """A minimum implementation of a worker that executes a PyTorch model"""

    @staticmethod
    def deserialize(data_blob: bytes) -> InferenceRequest:
        request: InferenceRequest = pickle.loads(data_blob)
        return request

    @staticmethod
    def load_model(
        request: InferenceRequest, fetch_result: FetchModelResult
    ) -> ModelLoadResult:
        model_bytes = fetch_result.model_bytes or request.raw_model
        if not model_bytes:
            raise ValueError("Unable to load model without reference object")

        model: torch.nn.Module = torch.load(io.BytesIO(model_bytes))
        result = ModelLoadResult(model)
        return result

    @staticmethod
    def transform_input(
        request: InferenceRequest, fetch_result: InputFetchResult
    ) -> InputTransformResult:
        result = [torch.load(io.BytesIO(item)) for item in fetch_result.inputs]
        return InputTransformResult(result)
        # return data # note: this fails copy test!

    @staticmethod
    def execute(
        request: InferenceRequest,
        load_result: ModelLoadResult,
        transform_result: InputTransformResult,
    ) -> ExecuteResult:
        """Execute an ML model on the given inputs"""
        if not load_result.model:
            raise sse.SmartSimError("Model must be loaded to execute")

        model = load_result.model
        results = [model(tensor) for tensor in transform_result.transformed]

        execute_result = ExecuteResult(results)
        return execute_result

    @staticmethod
    def transform_output(
        request: InferenceRequest,
        execute_result: ExecuteResult,
    ) -> OutputTransformResult:
        transformed = [item.clone() for item in execute_result.predictions]
        return OutputTransformResult(transformed)

    @staticmethod
    def serialize_reply(reply: InferenceReply) -> t.Any:
        return pickle.dumps(reply)


class IntegratedTorchWorker(MachineLearningWorkerBase):
    """A minimum implementation of a worker that executes a PyTorch model"""

    @staticmethod
    def deserialize(data_blob: bytes) -> InferenceRequest:
        # todo: consider moving to XxxCore and only making
        # workers implement the inputs and model conversion?

        # alternatively, consider passing the capnproto models
        # to this method instead of the data_blob...

        # something is definitely wrong here... client shouldn't have to touch
        # callback (or batch size)

        request = MessageHandler.deserialize_request(data_blob)
        # return request
        device = None
        if request.device.which() == "deviceType":
            device = request.device.deviceType

        model_key: t.Optional[str] = None
        model_bytes: t.Optional[bytes] = None

        if request.model.which() == "modelKey":
            model_key = request.model.modelKey
        elif request.model.which() == "modelData":
            model_bytes = request.model.modelData

        callback_key = request.replyChannel.reply

        # todo: shouldn't this be `CommChannel.find` instead of `DragonCommChannel`
        comm_channel = DragonCommChannel.find(callback_key)
        # comm_channel = DragonCommChannel(request.replyChannel)

        input_keys: t.Optional[t.List[str]] = None
        input_bytes: t.Optional[t.List[bytes]] = (
            None  # these will really be tensors already
        )

        if request.input.which() == "inputKeys":
            input_keys = request.input.inputKeys
        elif request.input.which() == "inputData":
            input_bytes = request.input.inputData

        inf_req = InferenceRequest(
            model_key=model_key,
            callback=comm_channel,
            raw_inputs=input_bytes,
            input_keys=input_keys,
            raw_model=model_bytes,
            batch_size=0,
            device=device,
        )
        return inf_req

    @staticmethod
    def load_model(
        request: InferenceRequest, fetch_result: FetchModelResult
    ) -> ModelLoadResult:
        model_bytes = fetch_result.model_bytes or request.raw_model
        if not model_bytes:
            raise ValueError("Unable to load model without reference object")

        model: torch.nn.Module = torch.load(io.BytesIO(model_bytes))
        result = ModelLoadResult(model)
        return result

    @staticmethod
    def transform_input(
        request: InferenceRequest, fetch_result: InputFetchResult
    ) -> InputTransformResult:
        result = [torch.load(io.BytesIO(item)) for item in fetch_result.inputs]
        return InputTransformResult(result)
        # return data # note: this fails copy test!

    @staticmethod
    def execute(
        request: InferenceRequest,
        load_result: ModelLoadResult,
        transform_result: InputTransformResult,
    ) -> ExecuteResult:
        """Execute an ML model on the given inputs"""
        if not load_result.model:
            raise sse.SmartSimError("Model must be loaded to execute")

        model = load_result.model
        results = [model(tensor) for tensor in transform_result.transformed]

        execute_result = ExecuteResult(results)
        return execute_result

    @staticmethod
    def transform_output(
        request: InferenceRequest,
        execute_result: ExecuteResult,
    ) -> OutputTransformResult:
        transformed = [item.clone() for item in execute_result.predictions]
        return OutputTransformResult(transformed)

    @staticmethod
    def _prepare_outputs(reply: InferenceReply) -> t.List[t.Any]:
        results = []
        if reply.output_keys:
            for key in reply.output_keys:
                if not key:
                    continue
                msg_key = MessageHandler.build_tensor_key(key)
                results.append(msg_key)
        elif reply.outputs:
            for output in reply.outputs:
                tensor: torch.Tensor = output
                # todo: need to have the output attributes specified in the req?
                # maybe, add `MessageHandler.dtype_of(tensor)`?
                # can `build_tensor` do dtype and shape?
                msg_tensor = MessageHandler.build_tensor(
                    tensor,
                    "c",
                    "float32",
                    [1],
                )
                results.append(msg_tensor)
        return results

    @staticmethod
    def serialize_reply(reply: InferenceReply) -> t.Any:
        # todo: consider moving to XxxCore and only making
        # workers implement results-to-bytes
        if reply.failed:
            return MessageHandler.build_response(
                status=400,  # todo: need to indicate correct status
                message="fail",  # todo: decide what these will be
                result=[],
                custom_attributes=None,
            )

        results = IntegratedTorchWorker._prepare_outputs(reply)

        response = MessageHandler.build_response(
            status=200,  # todo: are we satisfied with 0/1 (success, fail)
            # todo: if not detailed messages, this shouldn't be returned.
            message="success",
            result=results,
            custom_attributes=None,
        )
        serialized_resp = MessageHandler.serialize_response(response)
        return serialized_resp
