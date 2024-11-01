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
import textwrap
import typing as t
from os import path as osp

from .._core.utils.helpers import create_short_id_str
from ..launchable import Job
from ..log import get_logger
from ..settings.launch_command import LauncherType
from ..settings.launch_settings import LaunchSettings

logger = get_logger(__name__)


# TODO: Remove this supression when we strip fileds/functionality
#       (run-settings/batch_settings/params_as_args/etc)!
# pylint: disable-next=too-many-public-methods


class InfrastructureService(abc.ABC):
    """The InfrastructureService class is an abstract class
    which defines how user-facing services (such as those used to run ML components)
    can be defined and launched in workflows.

    Infastructure services are only compatible with the DragonLauncher.

    """

    def __init__(
        self,
        identifier: str | None,
        num_nodes: int | None,
        hostnames: list[str] | None,
    ) -> None:
        """Initialize an ``InfrastructureService``

        Infrastructure services require a name and proper launch arguments.

        If both ``num_nodes`` and ``hostnames`` are specified and the length of
        ``hostnames`` is smaller than ``num_nodes``, the remaining hosts will be picked
        randomly among the available ones. If the length of ``hostnames`` is larger
        than ``num_nodes``, ``num_nodes`` nodes will be chosen among the ones
        contained in ``hostnames``.

        :param identifier: identifier which can be used by client apps, must be unique.
        across all infrastructure services; if one is not provided, a unique identifier
        is created.
        :param num_nodes: number of nodes on which the service needs to be launched
        :param hostnames: the host on which the service needs to be launched
        :param launch_settings: launch settings defining how the service will run
        :raises ValueError: if the launcher of launch_settings is not Dragon.
        """

        self._identifier = identifier or create_short_id_str()
        """Unique identifier of this service"""
        self._num_nodes = num_nodes
        """Number of nodes on which the service needs to be launched"""
        self._hostnames = copy.deepcopy(hostnames)
        """Host(s) on which the service needs to be launched"""

    @property
    def identifier(self):
        return self._identifier

    def __str__(self) -> str:  # pragma: no cover

        return textwrap.dedent(
            f"""\
            Identifier: {self.identifier}
            Type: {self.type}
            """
        )

    @abc.abstractmethod
    def _build_exe_args(self) -> list[str]: ...
