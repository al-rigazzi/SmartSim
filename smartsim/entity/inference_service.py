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

from __future__ import annotations

import abc
import collections
import copy
import sys
import textwrap
import typing as t
from os import path as osp

from .._core.generation.operations.operations import FileSysOperationSet
from .._core.utils.helpers import create_short_id_str, expand_exe_path
from ..error import SSUnsupportedError
from ..launchable import Job
from ..log import get_logger
from ..settings.launch_command import LauncherType
from ..settings.launch_settings import DragonLaunchArguments, LaunchSettings
from . import SmartSimEntity
from .application import Application
from .infrastructure_service import InfrastructureService

logger = get_logger(__name__)


# TODO: Remove this supression when we strip fileds/functionality
#       (run-settings/batch_settings/params_as_args/etc)!
# pylint: disable-next=too-many-public-methods


class InferenceService(InfrastructureService, abc.ABC):
    """The InferenceService class is an abstract class
    which defines how user-facing inference services
    can be defined and launched in workflows.

    Infastructure services are only compatible with the DragonLauncher.

    """

    def __init__(
        self,
        identifier: str | None,
        launch_settings: LaunchSettings,
        device: t.Literal["gpu", "cpu"] = "cpu",
        num_workers: int = 1,
        batch_size: int = 1,
        batch_timeout: float = 0.0,
        toolkit: str = "",
    ) -> None:
        """Initialize an ``InferenceService``

        Each InferenceService is currently limited to run on one node only, using
        ``LaunchSettings`` with more than one node specified will raise an error.
        To take advantage of vectorization, multiple requests can be batched together,
        i.e. the service can wait until a ``batch_size`` requests are received before
        executing them all as one single inference call. Batches which are not complete
        can also be executed after ``batch_timeout`` seconds elapse.


        :param identifier: identifier which can be used by client apps, must be unique
        across all infrastructure services; if one is not provided, a unique identifier
        is created.
        :param launch_settings: launch settings defining how the service will run.
        :param device: Device to use for inference, can be "cpu" or "gpu".
        :param num_workers: Number of workers that should serve requests. If the
        ``device`` is "gpu", ``num_workers`` should be less or equal to the number
        of available GPUs.
        :param batch_size: how many *requests* should be batched together before
        running inference.
        :param batch_timeout: how long (in seconds) the service should wait before
        running inference on an incomplete batch.
        :param toolkit: the toolkit to use to run inference.
        :raises ValueError: if the launcher of launch_settings is not Dragon.
        :raises SSUnsupportedError: if ``launch_arguments`` specifies a number of nodes
        greater than one.
        """
        super().__init__(identifier=identifier, launch_settings=launch_settings)
        self._device = device
        """Device to use for inference"""
        self._num_workers = num_workers
        """Number of workers that should serve requests"""
        self._batch_size = batch_size
        """How many requests should be batched together before running inference."""
        self._batch_timeout = batch_timeout
        """How long (in seconds) the service should wait before running inference on
        an incomplete batch"""
        self._toolkit = toolkit
        """The type of worker to run"""

    @property
    def launch_settings(self) -> LaunchSettings:
        """Return the launch settings.

        :return: the launch settings
        """
        return self._launch_settings

    @launch_settings.setter
    def launch_settings(self, value: LaunchSettings) -> None:
        """Set the launch arguments.

        :param value: the launch arguments.
        :raises ValueError: if the launcher of launch_settings is not Dragon.
        :raises ValueError: if more than one node is requested by value
        """
        if value.launcher != LauncherType.Dragon.value:
            raise ValueError(
                "Infrastructure services can only be run with Dragon"
                f" launcher, but {value.launcher} was supplied."
            )

        if "nodes" in value.launch_args._launch_args:
            requested_nodes = value.launch_args._launch_args.get("nodes", 1)

            if requested_nodes is not None and requested_nodes != 1:
                raise SSUnsupportedError(
                    f"{type(self).__name__} can only be launched on one"
                    f"node, but {requested_nodes} nodes were requested"
                )

        self._launch_settings = copy.deepcopy(value)
        """Launch settings"""

    def __str__(self) -> str:  # pragma: no cover

        return textwrap.dedent(f"""\
            Identifier: {self.name}
            Type: {self.type}
            """)

    def _build_exe_args(self) -> list[str]:
        exe_args = [
            "-m",
            "smartsim._core.entrypoints.inference_service",
            "--device",
            self._device,
            "--toolkit",
            self._toolkit,
            "--num_workers",
            str(self._num_workers),
            "--batch_size",
            str(self._batch_size),
            "--batch_timeout",
            str(self._batch_timeout),
            "--identifier",
            self.name,
        ]
        return exe_args

    def build_jobs(self) -> list[Job]:
        """Build and return jobs needed to run the services

        :return: A list of jobs to launch run this service"""
        exe_args = self._build_exe_args()

        app = Application(
            name=self.name,
            exe=sys.executable,
            exe_args=exe_args,
        )

        job = Job(app, launch_settings=self._launch_settings)

        return [job]

    def as_executable_sequence(self) -> t.Sequence[str]:
        """Converts the executable and its arguments into a sequence of program arguments.

        :return: a sequence of strings representing the executable and its arguments
        """
        return [sys.executable, *self._build_exe_args()]


class TorchInferenceService(InferenceService):
    """The TorchInferenceService adds Torch-based inference capabilities
    to a workflow.

    Infastructure services are only compatible with the DragonLauncher.

    """

    def __init__(
        self,
        identifier: str | None,
        launch_settings: LaunchSettings,
        device: t.Literal["gpu", "cpu"] = "cpu",
        num_workers: int = 1,
        batch_size: int = 1,
        batch_timeout: float = 0.0,
    ) -> None:
        """Initialize a ``TorchInfrastructureService``

        Each ``TorchInferenceService`` is currently limited to run on one node only,
        using ``LaunchSettings`` with more than one node specified will raise
        an error.
        To take advantage of vectorization, multiple requests can be batched together,
        i.e. the service can wait until a ``batch_size`` requests are received before
        executing them all as one single inference call. Batches which are not complete
        can also be executed after ``batch_timeout`` seconds elapse.


        param identifier: identifier which can be used by client apps, must be unique
        across all infrastructure services; if one is not provided, a unique identifier
        is created.
        :param launch_settings: launch settings defining how the service will run.
        :param device: Device to use for inference, can be "cpu" or "gpu".
        :param num_workers: Number of workers that should serve requests. If the
        ``device`` is "gpu", ``num_workers`` should be less or equal to the number
        of available GPUs.
        :param batch_size: how many *requests* should be batched together before
        running inference.
        :param batch_timeout: how long (in seconds) the service should wait before
        running inference on an incomplete batch.
        :raises ValueError: if the launcher of launch_settings is not Dragon.
        :raises SSUnsupportedError: if ``launch_arguments`` specifies a number of nodes
        greater than one.
        """
        super().__init__(
            identifier=identifier,
            launch_settings=launch_settings,
            device=device,
            num_workers=num_workers,
            batch_size=batch_size,
            batch_timeout=batch_timeout,
            toolkit="torch",
        )


class TensorFlowInferenceService(InferenceService):
    """The TensorFlowInferenceService adds TensorFlow-based inference capabilities
    to a workflow.

    Infastructure services are only compatible with the DragonLauncher.

    """

    def __init__(
        self,
        identifier: str | None,
        launch_settings: LaunchSettings,
        device: t.Literal["gpu", "cpu"] = "cpu",
        num_workers: int = 1,
        batch_size: int = 1,
        batch_timeout: float = 0.0,
    ) -> None:
        """Initialize a ``TensorFlowInfrastructureService``

        Each ``TensorFlowInferenceService`` is currently limited to run on one node
        only, using ``LaunchSettings`` with more than one node specified will
        raise an error.
        To take advantage of vectorization, multiple requests can be batched together,
        i.e. the service can wait until a ``batch_size`` requests are received before
        executing them all as one single inference call. Batches which are not complete
        can also be executed after ``batch_timeout`` seconds elapse.


        param identifier: identifier which can be used by client apps, must be unique
        across all infrastructure services; if one is not provided, a unique identifier
        is created.
        :param launch_settings: launch settings defining how the service will run.
        :param device: Device to use for inference, can be "cpu" or "gpu".
        :param num_workers: Number of workers that should serve requests. If the
        ``device`` is "gpu", ``num_workers`` should be less or equal to the number
        of available GPUs.
        :param batch_size: how many *requests* should be batched together before
        running inference.
        :param batch_timeout: how long (in seconds) the service should wait before
        running inference on an incomplete batch.
        :raises ValueError: if the launcher of launch_settings is not Dragon.
        :raises SSUnsupportedError: if ``launch_arguments`` specifies a number of nodes
        greater than one.
        """
        super().__init__(
            identifier=identifier,
            launch_settings=launch_settings,
            device=device,
            num_workers=num_workers,
            batch_size=batch_size,
            batch_timeout=batch_timeout,
            toolkit="tensorflow",
        )


class ONNXInferenceService(InferenceService):
    """The ONNXInferenceService adds ONNX-based inference capabilities
    to a workflow.

    Infastructure services are only compatible with the DragonLauncher.

    """

    def __init__(
        self,
        identifier: str | None,
        launch_settings: LaunchSettings,
        device: t.Literal["gpu", "cpu"] = "cpu",
        num_workers: int = 1,
        batch_size: int = 1,
        batch_timeout: float = 0.0,
    ) -> None:
        """Initialize a ``ONNXInferenceService``

        Each ``InferenceService`` is currently limited to run on one node only,
        using ``LaunchSettings`` with more than one node specified will raise
        an error.
        To take advantage of vectorization, multiple requests can be batched together,
        i.e. the service can wait until a ``batch_size`` requests are received before
        executing them all as one single inference call. Batches which are not complete
        can also be executed after ``batch_timeout`` seconds elapse.


        param identifier: identifier which can be used by client apps, must be unique
        across all infrastructure services; if one is not provided, a unique identifier
        is created.
        :param launch_settings: launch settings defining how the service will run.
        :param device: Device to use for inference, can be "cpu" or "gpu".
        :param num_workers: Number of workers that should serve requests. If the
        ``device`` is "gpu", ``num_workers`` should be less or equal to the number
        of available GPUs.
        :param batch_size: how many *requests* should be batched together before
        running inference.
        :param batch_timeout: how long (in seconds) the service should wait before
        running inference on an incomplete batch.
        :raises ValueError: if the launcher of launch_settings is not Dragon.
        :raises SSUnsupportedError: if ``launch_arguments`` specifies a number of nodes
        greater than one.
        """
        super().__init__(
            identifier=identifier,
            launch_settings=launch_settings,
            device=device,
            num_workers=num_workers,
            batch_size=batch_size,
            batch_timeout=batch_timeout,
            toolkit="onnx",
        )
