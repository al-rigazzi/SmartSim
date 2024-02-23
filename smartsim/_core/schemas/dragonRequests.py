# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

import typing as t
import abc

from pydantic import BaseModel, PositiveInt

from smartsim._core.schemas.types import NonEmptyStr


class DragonRequest(BaseModel, abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def type() -> str: ...


class DragonRunRequestView(DragonRequest):
    exe: NonEmptyStr
    exe_args: t.List[NonEmptyStr] = []
    path: NonEmptyStr
    nodes: PositiveInt = 1
    tasks: PositiveInt = 1
    output_file: t.Optional[NonEmptyStr] = None
    error_file: t.Optional[NonEmptyStr] = None
    env: t.Dict[str, t.Optional[str]] = {}
    name: t.Optional[NonEmptyStr]
    pmi_enabled: bool = True

    @staticmethod
    def type() -> str:
        return "run"


class DragonRunRequest(DragonRunRequestView):
    current_env: t.Dict[str, t.Optional[str]] = {}

    def __str__(self) -> str:
        return str(DragonRunRequestView.parse_obj(self.dict(exclude={"current_env"})))


class DragonUpdateStatusRequest(DragonRequest):
    step_ids: t.List[NonEmptyStr]

    @staticmethod
    def type() -> str:
        return "update_status"


class DragonStopRequest(DragonRequest):
    step_id: NonEmptyStr

    @staticmethod
    def type() -> str:
        return "stop"


class DragonHandshakeRequest(DragonRequest):
    @staticmethod
    def type() -> str:
        return "handshake"


class DragonBootstrapRequest(DragonRequest):
    address: NonEmptyStr

    @staticmethod
    def type() -> str:
        return "bootstrap"
