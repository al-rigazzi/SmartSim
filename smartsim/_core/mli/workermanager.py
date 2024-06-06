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

import logging
import multiprocessing as mp
import pathlib
import time
import typing as t
import uuid

import torch

from smartsim._core.mli.util import ServiceHost
from smartsim._core.mli.worker import IntegratedTorchWorker
from smartsim.log import get_logger

from .infrastructure import DragonCommChannel, FeatureStore, MemoryFeatureStore
from .message_handler import MessageHandler
from .worker import InferenceReply, MachineLearningWorkerBase

if t.TYPE_CHECKING:
    import dragon.channels as dch
    import dragon.utils as du


logger = get_logger(__name__)


class WorkerManager(ServiceHost):
    """An implementation of a service managing distribution of tasks to
    machine learning workers"""

    def __init__(
        self,
        feature_store: FeatureStore,
        worker: MachineLearningWorkerBase,
        as_service: bool = False,
        cooldown: int = 0,
    ) -> None:
        """Initialize the WorkerManager
        :param feature_store: The persistence mechanism
        :param worker: A worker to manage
        :param as_service: Specifies run-once or run-until-complete behavior of service
        :param cooldown: Number of seconds to wait before shutting down afer
        shutdown criteria are met"""
        super().__init__(as_service, cooldown)

        self._workers: t.Dict[
            str, "t.Tuple[MachineLearningWorkerBase, mp.Queue[bytes]]"
        ] = {}
        """a collection of workers the manager is controlling"""
        self._upstream_queue: t.Optional[mp.Queue[bytes]] = None
        """the queue the manager monitors for new tasks"""
        self._feature_store: FeatureStore = feature_store
        """a feature store to retrieve models from"""
        self._worker = worker
        """The ML Worker implementation"""

    @property
    def upstream_queue(self) -> "t.Optional[mp.Queue[bytes]]":
        """Return the queue used by the worker manager to receive new work"""
        return self._upstream_queue

    @upstream_queue.setter
    def upstream_queue(self, value: "mp.Queue[bytes]") -> None:
        """Set/update the queue used by the worker manager to receive new work"""
        self._upstream_queue = value

    def _on_iteration(self, timestamp: int) -> None:
        """Executes calls to the machine learning worker implementation to complete
        the inference pipeline"""
        logger.debug(f"{timestamp} executing worker manager pipeline")

        if self.upstream_queue is None:
            logger.warning("No queue to check for tasks")
            return

        msg: bytes = self.upstream_queue.get()
        request = self._worker.deserialize(msg)
        fetch_model_result = self._worker.fetch_model(request, self._feature_store)
        model_result = self._worker.load_model(request, fetch_model_result)
        fetch_input_result = self._worker.fetch_inputs(
            request,
            self._feature_store,
        )  # we don't know if they'lll fetch in some weird way
        # they potentially need access to custom attributes
        # we don't know what the response really is... i have it as bytes
        # but we just want to advertise that the contract states "the output
        # will be the input to transform_input... "

        transform_result = self._worker.transform_input(request, fetch_input_result)

        # batch: t.Collection[_Datum] = transform_result.transformed_input
        # if self._batch_size:
        #     batch = self._worker.batch_requests(transform_result, self._batch_size)

        # todo: what do we return here? tensors? Datum? bytes?
        results = self._worker.execute(request, model_result, transform_result)

        # todo: inference reply _must_ be replaced with the mli schemas reply
        reply = InferenceReply()

        # only place into feature store if keys are provided
        if request.output_keys:
            output_keys = self._worker.place_output(
                request, results, self._feature_store
            )
            reply.output_keys = output_keys
        else:
            reply.outputs = results.predictions

        serialized_output = self._worker.serialize_reply(reply)

        callback_channel = request.callback
        if callback_channel:
            callback_channel.send(serialized_output)

    def _can_shutdown(self) -> bool:
        """Return true when the criteria to shut down the service are met."""
        return bool(self._workers)

    def add_worker(
        self, worker: MachineLearningWorkerBase, work_queue: "mp.Queue[bytes]"
    ) -> None:
        """Add a worker instance to the collection managed by the WorkerManager"""
        self._workers[str(uuid.uuid4())] = (worker, work_queue)


def mock_work(worker_manager_queue: "mp.Queue[bytes]") -> None:
    """Mock event producer for triggering the inference pipeline"""
    # todo: move to unit tests
    while True:
        time.sleep(1)
        # 1. for demo, ignore upstream and just put stuff into downstream
        # 2. for demo, only one downstream but we'd normally have to filter
        #       msg content and send to the correct downstream (worker) queue
        timestamp = time.time_ns()
        work_dir = "/lus/bnchlu1/mcbridch/code/ss/tests/test_output/brainstorm"
        test_path = pathlib.Path(work_dir)

        mock_channel = test_path / f"brainstorm-{timestamp}.txt"
        mock_model = test_path / "brainstorm.pt"

        test_path.mkdir(parents=True, exist_ok=True)
        mock_channel.touch()
        mock_model.touch()

        msg = f"PyTorch:{mock_model}:MockInputToReplace:{mock_channel}"
        worker_manager_queue.put(msg.encode("utf-8"))


def persist_model_file(work_dir: str) -> pathlib.Path:
    """Create a simple torch model and persist to disk for
    testing purposes.

    TODO: remove once unit tests are in place"""
    test_path = pathlib.Path(work_dir)
    model_path = test_path / "basic.pt"

    model = torch.nn.Linear(2, 1)
    torch.save(model, model_path)

    return model_path


def mock_messages(
    worker_manager_queue: "mp.Queue[bytes]", feature_store: FeatureStore, test_dir: str
) -> None:
    """Mock event producer for triggering the inference pipeline"""
    # todo: move to unit tests
    model_key = "persisted-model"
    model_bytes = persist_model_file(test_dir).read_bytes()
    feature_store[model_key] = model_bytes

    iteration_number = 0

    while True:
        iteration_number += 1
        time.sleep(1)
        # 1. for demo, ignore upstream and just put stuff into downstream
        # 2. for demo, only one downstream but we'd normally have to filter
        #       msg content and send to the correct downstream (worker) queue
        timestamp = time.time_ns()
        test_dir = "/lus/bnchlu1/mcbridch/tmp"
        test_path = pathlib.Path(test_dir)
        test_path.mkdir(parents=True, exist_ok=True)

        mock_channel = test_path / f"brainstorm-{timestamp}.txt"
        mock_model = test_path / "brainstorm.pt"

        test_path.mkdir(parents=True, exist_ok=True)
        mock_channel.touch()
        mock_model.touch()

        # thread - just look for key (wait for keys)
        # call checkpoint, try to get non-persistent key, it blocks
        # working set size > 1 has side-effects
        # only incurs cost when working set size has been exceeded

        # msg = f"PyTorch:{mock_model}:MockInputToReplace:{mock_channel}"
        # input_tensor = torch.randn(2)

        expected_device = "cpu"
        channel_key = b"faux_channel_descriptor_bytes"
        callback_channel = DragonCommChannel.find(channel_key)

        input_key = f"demo-{iteration_number}"
        output_key = f"demo-{iteration_number}-out"

        # feature_store[input_key] = input_tensor

        # message_input_tensor =
        # MessageHandler.build_tensor(input_tensor, "c", "float32", [2])
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
        worker_manager_queue.put(MessageHandler.serialize_request(request))


if __name__ == "__main__":
    logging.basicConfig(filename="workermanager.log")
    # queue for communicating to the worker manager. used to
    # simulate messages "from the application"
    upstream_queue: "mp.Queue[bytes]" = mp.Queue()

    # downstream_queue = mp.Queue()  # the queue to forward messages to a given worker

    # torch_worker = SampleTorchWorker(downstream_queue)
    integrated_worker = IntegratedTorchWorker()

    mem_fs = MemoryFeatureStore()

    worker_manager = WorkerManager(
        mem_fs, integrated_worker, as_service=True, cooldown=10
    )
    # worker_manager = WorkerManager(dict_fs, as_service=True, cooldown=10)
    # # configure what the manager listens to
    # worker_manager.upstream_queue = upstream_queue
    # # # and configure a worker ... moving...
    # # will dynamically add a worker in the manager based on input msg backend
    # # worker_manager.add_worker(torch_worker, downstream_queue)

    # # create a pretend to populate the queues
    # msg_pump = mp.Process(target=mock_work, args=(upstream_queue,))
    msg_pump = mp.Process(
        target=mock_messages, args=(upstream_queue, mem_fs, "/lus/bnchlu1/mcbridch/tmp")
    )
    msg_pump.start()

    # create a process to process commands
    process = mp.Process(target=worker_manager.execute, args=(time.time_ns(),))
    process.start()
    process.join()

    msg_pump.kill()
    # logger.info(f"{DefaultTorchWorker.backend()=}")
