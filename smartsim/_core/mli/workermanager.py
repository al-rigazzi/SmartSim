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
import logging
import multiprocessing as mp
import pathlib
import shutil
import time
import typing as t

import torch

from smartsim._core.mli.infrastructure import (
    CommChannelBase,
    DragonCommChannel,
    FeatureStore,
    FileSystemCommChannel,
    FileSystemFeatureStore,
)
from smartsim._core.mli.message_handler import MessageHandler
from smartsim._core.mli.mli_schemas.response.response_capnp import Response
from smartsim._core.mli.util import Service
from smartsim._core.mli.worker import (
    InferenceReply,
    InferenceRequest,
    IntegratedTorchWorker,
    MachineLearningWorkerBase,
)
from smartsim.log import get_logger

if t.TYPE_CHECKING:
    import dragon.channels as dch
    import dragon.utils as du


logger = get_logger(__name__)


def deserialize_message(
    data_blob: bytes, channel_type: t.Type[CommChannelBase]
) -> InferenceRequest:
    """Deserialize a message from a byte stream into an InferenceRequest
    :param data_blob: The byte stream to deserialize"""
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
        model_key = request.model.modelKey.key
    elif request.model.which() == "modelData":
        model_bytes = request.model.modelData

    callback_key = request.replyChannel.reply

    # todo: shouldn't this be `CommChannel.find` instead of `DragonCommChannel`
    comm_channel = channel_type.find(callback_key)
    # comm_channel = DragonCommChannel(request.replyChannel)

    input_keys: t.Optional[t.List[str]] = None
    input_bytes: t.Optional[t.List[bytes]] = (
        None  # these will really be tensors already
    )

    # # client example
    # msg = Message()
    # t = torch.Tensor()
    # msg.inputs = [custom_byte_converter(t)]
    # mli_client.request_inference(msg)
    # # end client
    input_meta: t.List[t.Any] = []

    if request.input.which() == "inputKeys":
        input_keys = [input_key.key for input_key in request.input.inputKeys]
    elif request.input.which() == "inputData":
        input_bytes = [data.blob for data in request.input.inputData]
        input_meta = [data.tensorDescriptor for data in request.input.inputData]

    inference_request = InferenceRequest(
        model_key=model_key,
        callback=comm_channel,
        raw_inputs=input_bytes,
        input_meta=input_meta,
        input_keys=input_keys,
        raw_model=model_bytes,
        batch_size=0,
        device=device,
    )
    return inference_request


def build_failure_reply(status: int, message: str) -> Response:
    return MessageHandler.build_response(
        status=status,  # todo: need to indicate correct status
        message=message,  # todo: decide what these will be
        result=[],
        custom_attributes=None,
    )


def prepare_outputs(reply: InferenceReply) -> t.List[t.Any]:
    prepared_outputs: t.List[t.Any] = []
    if reply.output_keys:
        for key in reply.output_keys:
            if not key:
                continue
            msg_key = MessageHandler.build_tensor_key(key)
            prepared_outputs.append(msg_key)
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
            prepared_outputs.append(msg_tensor)
    return prepared_outputs


def build_reply(reply: InferenceReply) -> Response:
    results = prepare_outputs(reply)

    return MessageHandler.build_response(
        status=200,
        message="success",
        result=results,
        custom_attributes=None,
    )


class WorkerManager(Service):
    """An implementation of a service managing distribution of tasks to
    machine learning workers"""

    def __init__(
        self,
        feature_store: FeatureStore,
        worker: MachineLearningWorkerBase,
        task_queue: "mp.Queue[bytes]",
        as_service: bool = False,
        cooldown: int = 0,
        comm_channel_type: t.Type[CommChannelBase] = DragonCommChannel,
    ) -> None:
        """Initialize the WorkerManager
        :param feature_store: The persistence mechanism
        :param workers: A worker to manage
        :param as_service: Specifies run-once or run-until-complete behavior of service
        :param cooldown: Number of seconds to wait before shutting down afer
        shutdown criteria are met"""
        super().__init__(as_service, cooldown)

        """a collection of workers the manager is controlling"""
        self._task_queue: "mp.Queue[bytes]" = task_queue
        """the queue the manager monitors for new tasks"""
        self._feature_store: FeatureStore = feature_store
        """a feature store to retrieve models from"""
        self._worker = worker
        """The ML Worker implementation"""
        self._comm_channel_type = comm_channel_type
        """The type of communication channel to construct for callbacks"""

    def _on_iteration(self) -> None:
        """Executes calls to the machine learning worker implementation to complete
        the inference pipeline"""
        logger.debug("executing worker manager pipeline")

        if self._task_queue is None:
            logger.warning("No queue to check for tasks")
            return

        # perform default deserialization of the message envelope
        request_bytes: bytes = self._task_queue.get()

        request = deserialize_message(request_bytes, self._comm_channel_type)
        if not request.callback:
            logger.error("No callback channel provided in request")
            return

        # # let the worker perform additional custom deserialization
        # request = self._worker.deserialize(request_bytes)

        fetch_model_result = self._worker.fetch_model(request, self._feature_store)
        model_result = self._worker.load_model(request, fetch_model_result)

        fetch_input_result = self._worker.fetch_inputs(request, self._feature_store)
        transformed_input = self._worker.transform_input(request, fetch_input_result)

        # batch: t.Collection[_Datum] = transform_result.transformed_input
        # if self._batch_size:
        #     batch = self._worker.batch_requests(transform_result, self._batch_size)

        reply = InferenceReply()

        try:
            execute_result = self._worker.execute(
                request, model_result, transformed_input
            )

            transformed_output = self._worker.transform_output(request, execute_result)

            if request.output_keys:
                reply.output_keys = self._worker.place_output(
                    request, transformed_output, self._feature_store
                )
            else:
                reply.outputs = transformed_output.outputs
        except Exception:
            logger.exception("Error executing worker")
            reply.failed = True

        if reply.failed:
            response = build_failure_reply(400, "fail")
        else:
            if reply.outputs is None or not reply.outputs:
                response = build_failure_reply(401, "no-results")

            response = build_reply(reply)

        # serialized = self._worker.serialize_reply(request, transformed_output)
        serialized_resp = MessageHandler.serialize_response(response)
        request.callback.send(serialized_resp)

    def _can_shutdown(self) -> bool:
        """Return true when the criteria to shut down the service are met."""
        # todo: determine shutdown criteria
        # will we receive a completion message?
        # will we let MLI mgr just kill this?
        # time_diff = self._last_event - datetime.datetime.now()
        # if time_diff.total_seconds() > self._cooldown:
        #     return True
        # return False
        return self._worker is None


def mock_work(worker_manager_queue: "mp.Queue[bytes]") -> None:
    """Mock event producer for triggering the inference pipeline"""
    # todo: move to unit tests
    while True:
        time.sleep(1)
        # 1. for demo, ignore upstream and just put stuff into downstream
        # 2. for demo, only one downstream but we'd normally have to filter
        #       msg content and send to the correct downstream (worker) queue
        timestamp = time.time_ns()
        output_dir = "/lus/bnchlu1/mcbridch/code/ss/_tmp"
        output_path = pathlib.Path(output_dir)

        mock_channel = output_path / f"brainstorm-{timestamp}.txt"
        mock_model = output_path / "brainstorm.pt"

        output_path.mkdir(parents=True, exist_ok=True)
        mock_channel.touch()
        mock_model.touch()

        msg = f"PyTorch:{mock_model}:MockInputToReplace:{mock_channel}"
        worker_manager_queue.put(msg.encode("utf-8"))


def persist_model_file(model_path: pathlib.Path) -> pathlib.Path:
    """Create a simple torch model and persist to disk for
    testing purposes.

    TODO: remove once unit tests are in place"""
    # test_path = pathlib.Path(work_dir)
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)

    model_path.unlink(missing_ok=True)
    # model_path = test_path / "basic.pt"

    model = torch.nn.Linear(2, 1)
    torch.save(model, model_path)

    return model_path


def mock_messages(
    worker_manager_queue: "mp.Queue[bytes]",
    feature_store: FeatureStore,
    feature_store_root_dir: pathlib.Path,
    comm_channel_root_dir: pathlib.Path,
) -> None:
    """Mock event producer for triggering the inference pipeline"""
    # todo: move to unit tests
    feature_store_root_dir.mkdir(parents=True, exist_ok=True)
    comm_channel_root_dir.mkdir(parents=True, exist_ok=True)

    # model_name = "persisted-model"
    model_path = persist_model_file(feature_store_root_dir.parent / "model_original.pt")
    model_bytes = model_path.read_bytes()
    model_key = str(feature_store_root_dir / "model_fs.pt")

    feature_store[model_key] = model_bytes

    iteration_number = 0

    while True:
        iteration_number += 1
        time.sleep(1)
        # 1. for demo, ignore upstream and just put stuff into downstream
        # 2. for demo, only one downstream but we'd normally have to filter
        #       msg content and send to the correct downstream (worker) queue
        # timestamp = time.time_ns()
        # mock_channel = test_path / f"brainstorm-{timestamp}.txt"
        # mock_channel.touch()

        # thread - just look for key (wait for keys)
        # call checkpoint, try to get non-persistent key, it blocks
        # working set size > 1 has side-effects
        # only incurs cost when working set size has been exceeded

        expected_device: t.Literal["cpu", "gpu"] = "cpu"
        channel_key = comm_channel_root_dir / f"{iteration_number}/channel.txt"
        callback_channel = FileSystemCommChannel.find(str(channel_key).encode("utf-8"))

        input_path = feature_store_root_dir / f"{iteration_number}/input.pt"
        output_path = feature_store_root_dir / f"{iteration_number}/output.pt"

        input_key = str(input_path)
        output_key = str(output_path)

        buffer = io.BytesIO()
        tensor = torch.randn((1, 2), dtype=torch.float32)
        torch.save(tensor, buffer)
        feature_store[input_key] = buffer.getvalue()

        message_tensor_output_key = MessageHandler.build_tensor_key(output_key)
        message_tensor_input_key = MessageHandler.build_tensor_key(input_key)
        message_model_key = MessageHandler.build_model_key(model_key)

        request = MessageHandler.build_request(
            reply_channel=callback_channel.descriptor,
            model=message_model_key,
            device=expected_device,
            inputs=[message_tensor_input_key],
            outputs=[message_tensor_output_key],
            custom_attributes=None,
        )
        request_bytes = MessageHandler.serialize_request(request)
        worker_manager_queue.put(request_bytes)


if __name__ == "__main__":

    def prepare_environment() -> pathlib.Path:
        """Cleanup prior outputs to run demo repeatedly"""
        path = pathlib.Path("/lus/bnchlu1/mcbridch/code/ss/_tmp")
        if path.exists():
            shutil.rmtree(path)  # clean up prior results

        path.mkdir(parents=True)
        logging.basicConfig(filename=str(path / "workermanager.log"))
        return path

    test_path = prepare_environment()
    fs_path = test_path / "feature_store"
    comm_path = test_path / "comm_store"

    work_queue: "mp.Queue[bytes]" = mp.Queue()
    # torch_worker = SampleTorchWorker(downstream_queue)
    integrated_worker = IntegratedTorchWorker()
    file_system_store = FileSystemFeatureStore()

    worker_manager = WorkerManager(
        file_system_store,
        integrated_worker,
        work_queue,
        as_service=True,
        cooldown=10,
        comm_channel_type=FileSystemCommChannel,
    )

    # create a mock client application to populate the request queue
    # msg_pump = mp.Process(target=mock_work, args=(work_queue,))
    msg_pump = mp.Process(
        target=mock_messages,
        args=(work_queue, file_system_store, fs_path, comm_path),
    )
    msg_pump.start()

    # # create a process to process commands
    process = mp.Process(target=worker_manager.execute)
    process.start()
    process.join()
    msg_pump.kill()
    # logger.info(f"{DefaultTorchWorker.backend()=}")
