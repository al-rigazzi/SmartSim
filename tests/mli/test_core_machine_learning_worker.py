import pathlib
import time
import typing as t

import pytest
import torch

import smartsim.error as sse
from smartsim._core.mli import workermanager as mli
from smartsim._core.mli.infrastructure import FeatureStore, MemoryFeatureStore
from smartsim._core.mli.worker import (
    BatchResult,
    ExecuteResult,
    InferenceReply,
    InferenceRequest,
    InputTransformResult,
    MachineLearningWorkerCore,
    OutputTransformResult,
)
from smartsim._core.utils import installed_redisai_backends

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_b

# retrieved from pytest fixtures
is_dragon = pytest.test_launcher == "dragon"
torch_available = "torch" in installed_redisai_backends()


class FileSystemFeatureStore(FeatureStore):
    """Alternative feature store implementation for testing. Stores all
    data on the file system"""

    def __init__(self, storage_dir: t.Optional[pathlib.Path] = None) -> None:
        """Initialize the FileSystemFeatureStore instance
        :param storage_dir: (optional) root directory to store all data relative to"""
        self._storage_dir = storage_dir

    def __getitem__(self, key: str) -> bytes:
        """Retrieve an item using key
        :param key: Unique key of an item to retrieve from the feature store"""
        path = self._key_path(key)
        if not path.exists():
            raise sse.SmartSimError(f"{path} not found in feature store")
        return path.read_bytes()

    def __setitem__(self, key: str, value: bytes) -> None:
        """Assign a value using key
        :param key: Unique key of an item to set in the feature store
        :param value: Value to persist in the feature store"""
        path = self._key_path(key)
        path.write_bytes(value)

    def __contains__(self, key: str) -> bool:
        """Membership operator to test for a key existing within the feature store.
        Return `True` if the key is found, `False` otherwise
        :param key: Unique key of an item to retrieve from the feature store"""
        path = self._key_path(key)
        return path.exists()

    def _key_path(self, key: str) -> pathlib.Path:
        """Given a key, return a path that is optionally combined with a base
        directory used by the FileSystemFeatureStore.
        :param key: Unique key of an item to retrieve from the feature store"""
        if self._storage_dir:
            return self._storage_dir / key

        return pathlib.Path(key)


@pytest.fixture
def persist_torch_model(test_dir: str) -> pathlib.Path:
    ts_start = time.time_ns()
    print("Starting model file creation...")
    test_path = pathlib.Path(test_dir)
    model_path = test_path / "basic.pt"

    model = torch.nn.Linear(2, 1)
    torch.save(model, model_path)
    ts_end = time.time_ns()

    ts_elapsed = (ts_end - ts_start) / 1000000000
    print(f"Model file creation took {ts_elapsed} seconds")
    return model_path


@pytest.fixture
def persist_torch_tensor(test_dir: str) -> pathlib.Path:
    ts_start = time.time_ns()
    print("Starting model file creation...")
    test_path = pathlib.Path(test_dir)
    file_path = test_path / "tensor.pt"

    tensor = torch.randn((100, 100, 2))
    torch.save(tensor, file_path)
    ts_end = time.time_ns()

    ts_elapsed = (ts_end - ts_start) / 1000000000
    print(f"Tensor file creation took {ts_elapsed} seconds")
    return file_path


@pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
def test_fetch_model_disk(persist_torch_model: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a model
    when given a valid (file system) key"""
    worker = MachineLearningWorkerCore
    key = str(persist_torch_model)
    feature_store = FileSystemFeatureStore()
    feature_store[str(persist_torch_model)] = persist_torch_model.read_bytes()

    request = InferenceRequest(model_key=key)

    fetch_result = worker.fetch_model(request, feature_store)
    assert fetch_result.model_bytes
    assert fetch_result.model_bytes == persist_torch_model.read_bytes()


def test_fetch_model_disk_missing() -> None:
    """Verify that the ML worker fails to retrieves a model
    when given an invalid (file system) key"""
    worker = MachineLearningWorkerCore
    feature_store = MemoryFeatureStore()

    key = "/path/that/doesnt/exist"

    request = InferenceRequest(model_key=key)

    with pytest.raises(sse.SmartSimError) as ex:
        worker.fetch_model(request, feature_store)

    # ensure the error message includes key-identifying information
    assert key in ex.value.args[0]


@pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
def test_fetch_model_feature_store(persist_torch_model: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a model
    when given a valid (file system) key"""
    worker = MachineLearningWorkerCore

    # create a key to retrieve from the feature store
    key = "test-model"

    # put model bytes into the feature store
    feature_store = MemoryFeatureStore()
    feature_store[key] = persist_torch_model.read_bytes()

    request = InferenceRequest(model_key=key)
    fetch_result = worker.fetch_model(request, feature_store)
    assert fetch_result.model_bytes
    assert fetch_result.model_bytes == persist_torch_model.read_bytes()


def test_fetch_model_feature_store_missing() -> None:
    """Verify that the ML worker fails to retrieves a model
    when given an invalid (feature store) key"""
    worker = MachineLearningWorkerCore

    bad_key = "some-key"
    feature_store = MemoryFeatureStore()

    request = InferenceRequest(model_key=bad_key)

    # todo: consider that raising this exception shows impl. replace...
    with pytest.raises(sse.SmartSimError) as ex:
        worker.fetch_model(request, feature_store)

    # ensure the error message includes key-identifying information
    assert bad_key in ex.value.args[0]


@pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
def test_fetch_model_memory(persist_torch_model: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a model
    when given a valid (file system) key"""
    worker = MachineLearningWorkerCore

    key = "test-model"
    feature_store = MemoryFeatureStore()
    feature_store[key] = persist_torch_model.read_bytes()

    request = InferenceRequest(model_key=key)

    fetch_result = worker.fetch_model(request, feature_store)
    assert fetch_result.model_bytes
    assert fetch_result.model_bytes == persist_torch_model.read_bytes()


@pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
def test_fetch_input_disk(persist_torch_tensor: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a tensor/input
    when given a valid (file system) key"""
    tensor_name = str(persist_torch_tensor)

    request = InferenceRequest(input_keys=[tensor_name])
    worker = MachineLearningWorkerCore

    feature_store = MemoryFeatureStore()
    feature_store[tensor_name] = persist_torch_tensor.read_bytes()

    fetch_result = worker.fetch_inputs(request, feature_store)
    assert fetch_result.inputs is not None


def test_fetch_input_disk_missing() -> None:
    """Verify that the ML worker fails to retrieves a tensor/input
    when given an invalid (file system) key"""
    worker = MachineLearningWorkerCore

    key = "/path/that/doesnt/exist"
    feature_store = MemoryFeatureStore()

    request = InferenceRequest(input_keys=[key])

    # todo: consider that raising this exception shows impl. replace...
    with pytest.raises(sse.SmartSimError) as ex:
        worker.fetch_inputs(request, feature_store)

    # ensure the error message includes key-identifying information
    assert key in ex.value.args[0]


@pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
def test_fetch_input_feature_store(persist_torch_tensor: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a tensor/input
    when given a valid (feature store) key"""
    worker = MachineLearningWorkerCore

    tensor_name = "test-tensor"
    feature_store = MemoryFeatureStore()

    request = InferenceRequest(input_keys=[tensor_name])

    # put model bytes into the feature store
    feature_store[tensor_name] = persist_torch_tensor.read_bytes()

    fetch_result = worker.fetch_inputs(request, feature_store)
    assert fetch_result.inputs
    assert list(fetch_result.inputs)[0][:10] == persist_torch_tensor.read_bytes()[:10]


@pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
def test_fetch_multi_input_feature_store(persist_torch_tensor: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves multiple tensor/input
    when given a valid collection of (feature store) keys"""
    worker = MachineLearningWorkerCore

    tensor_name = "test-tensor"
    feature_store = MemoryFeatureStore()

    # put model bytes into the feature store
    body1 = persist_torch_tensor.read_bytes()
    feature_store[tensor_name + "1"] = body1

    body2 = b"abcdefghijklmnopqrstuvwxyz"
    feature_store[tensor_name + "2"] = body2

    body3 = b"mnopqrstuvwxyzabcdefghijkl"
    feature_store[tensor_name + "3"] = body3

    request = InferenceRequest(
        input_keys=[tensor_name + "1", tensor_name + "2", tensor_name + "3"]
    )

    fetch_result = worker.fetch_inputs(request, feature_store)

    raw_bytes = list(fetch_result.inputs)
    assert raw_bytes
    assert raw_bytes[0][:10] == persist_torch_tensor.read_bytes()[:10]
    assert raw_bytes[1][:10] == body2[:10]
    assert raw_bytes[2][:10] == body3[:10]


def test_fetch_input_feature_store_missing() -> None:
    """Verify that the ML worker fails to retrieves a tensor/input
    when given an invalid (feature store) key"""
    worker = MachineLearningWorkerCore

    bad_key = "some-key"
    feature_store = MemoryFeatureStore()
    request = InferenceRequest(input_keys=[bad_key])

    # todo: consider that raising this exception shows impl. replace...
    with pytest.raises(sse.SmartSimError) as ex:
        worker.fetch_inputs(request, feature_store)

    # ensure the error message includes key-identifying information
    assert bad_key in ex.value.args[0]


@pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
def test_fetch_input_memory(persist_torch_tensor: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a tensor/input
    when given a valid (file system) key"""
    worker = MachineLearningWorkerCore
    feature_store = MemoryFeatureStore()

    model_name = "test-model"
    feature_store[model_name] = persist_torch_tensor.read_bytes()
    request = InferenceRequest(input_keys=[model_name])

    fetch_result = worker.fetch_inputs(request, feature_store)
    assert fetch_result.inputs is not None


def test_batch_requests() -> None:
    """Verify batch requests handles an empty data set gracefully"""
    worker = MachineLearningWorkerCore
    result = InputTransformResult([])

    request = InferenceRequest(batch_size=10)

    with pytest.raises(NotImplementedError):
        # NOTE: we expect this to fail since it's not yet implemented.
        # TODO: once implemented, replace this expectation of failure...
        worker.batch_requests(request, result)


def test_place_outputs() -> None:
    """Verify outputs are shared using the feature store"""
    worker = MachineLearningWorkerCore

    key_name = "test-model"
    feature_store = MemoryFeatureStore()

    # create a key to retrieve from the feature store
    keys = [key_name + "1", key_name + "2", key_name + "3"]
    data = [b"abcdef", b"ghijkl", b"mnopqr"]

    for k, v in zip(keys, data):
        feature_store[k] = v

    request = InferenceRequest(output_keys=keys)
    execute_result = ExecuteResult(data)

    worker.place_output(request, execute_result, feature_store)

    for i in range(3):
        assert feature_store[keys[i]] == data[i]
