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
import typing as t
from abc import ABC, abstractmethod

import torch

import smartsim.error as sse
from smartsim.log import get_logger

from .infrastructure import CommChannel, FeatureStore

logger = get_logger(__name__)


class InferenceRequest:
    """Temporary model of an inference request"""

    def __init__(
        self,
        model_key: t.Optional[str] = None,
        callback: t.Optional[CommChannel] = None,
        raw_inputs: t.Optional[t.List[bytes]] = None,
        input_keys: t.Optional[t.List[str]] = None,
        input_meta: t.Optional[t.List[t.Any]] = None,
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
        self.input_meta = input_meta or []
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

    def __init__(self, result: t.List[bytes]) -> None:
        """Initialize the InputFetchResult"""
        self.inputs = result


class OutputTransformResult:
    """A wrapper around inference results transformed for transmission"""

    def __init__(
        self, result: t.Any, shape: t.List[int], order: str, dtype: str
    ) -> None:
        """Initialize the OutputTransformResult"""
        self.outputs = result
        self.shape = shape
        self.order = order
        self.dtype = dtype
        # todo: determine if each output must have an individual (shape, order, dtype)


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
        transform_result: OutputTransformResult,
        feature_store: FeatureStore,
    ) -> t.Collection[t.Optional[str]]:
        """Given a collection of data, make it available as a shared resource in the
        feature store
        :param request: The request that triggered the pipeline
        :param execute_result: Results from inference
        :param feature_store: The feature store used for persistence
        :return: A collection of keys that were placed in the feature store"""
        keys: t.List[t.Optional[str]] = []
        # need to decide how to get back to original sub-batch inputs so they can be
        # accurately placed, datum might need to include this.

        # Consider parallelizing all PUT feature_store operations
        for k, v in zip(request.output_keys, transform_result.transformed):
            feature_store[k] = v
            keys.append(k)

        return keys


class MachineLearningWorkerBase(MachineLearningWorkerCore, ABC):
    """Abstrct base class providing contract for a machine learning
    worker implementation."""

    # @staticmethod
    # @abstractmethod
    # def deserialize(request: InferenceRequest) -> InferenceRequest:
    #     """Given a collection of data serialized to bytes, convert the bytes
    #     to a proper representation used by the ML backend
    #     :param data_blob: inference request as a byte-serialized blob
    #     :return: InferenceRequest deserialized from the input"""

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

    # @staticmethod
    # @abstractmethod
    # def serialize_reply(
    #     request: InferenceRequest, results: OutputTransformResult
    # ) -> bytes:
    #     """Given an output, serialize to bytes for transport
    #     :param reply: The result of the inference pipeline
    #     :return: a byte-serialized version of the reply"""


class SampleTorchWorker(MachineLearningWorkerBase):
    """A minimum implementation of a worker that executes a PyTorch model"""

    @staticmethod
    def deserialize(request: InferenceRequest) -> InferenceRequest:
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
        # todo: need the shape from latest schemas added here.
        return OutputTransformResult(transformed, [1, 1, 1], "c", "float32")  # fixme

    # @staticmethod
    # def serialize_reply(
    #     request: InferenceRequest, results: OutputTransformResult
    # ) -> bytes:
    #     # return pickle.dumps(reply)
    #     return pickle.dumps(results.outputs)


class IntegratedTorchWorker(MachineLearningWorkerBase):
    """A minimum implementation of a worker that executes a PyTorch model"""

    # @staticmethod
    # def deserialize(request: InferenceRequest) -> t.List[t.Any]:
    #     # request.input_meta
    #     # request.raw_inputs
    #     return request

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
        request: InferenceRequest,
        fetch_result: InputFetchResult,
    ) -> InputTransformResult:
        # extra metadata for assembly can be found in request.input_meta
        raw_inputs = request.raw_inputs or fetch_result.inputs

        result: t.List[torch.Tensor] = []
        if raw_inputs:
            result = [torch.load(io.BytesIO(item)) for item in raw_inputs]

        return InputTransformResult(result)

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
        # transformed = [item.clone() for item in execute_result.predictions]
        # return OutputTransformResult(transformed)

        # transformed = [item.bytes() for item in execute_result.predictions]

        # OutputTransformResult.transformed SHOULD be a list of
        # capnproto Tensors Or tensor descriptors accompanying bytes

        # send the original tensors...
        execute_result.predictions = [t.detach() for t in execute_result.predictions]
        return OutputTransformResult(execute_result.predictions, [1], "c", "float32")
        # return OutputTransformResult(transformed)

    # @staticmethod
    # def serialize_reply(
    #     request: InferenceRequest, results: OutputTransformResult
    # ) -> t.Any:
    #     # results = IntegratedTorchWorker._prepare_outputs(results.outputs)
    #     # return results
    #     return None
    #     # response = MessageHandler.build_response(
    #     #     status=200,  # todo: are we satisfied with 0/1 (success, fail)
    #     #     # todo: if not detailed messages, this shouldn't be returned.
    #     #     message="success",
    #     #     result=results,
    #     #     custom_attributes=None,
    #     # )
    #     # serialized_resp = MessageHandler.serialize_response(response)
    #     # return serialized_resp
