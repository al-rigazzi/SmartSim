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

# pylint: disable-next=import-error
from dragon.native.process import Popen, Process

from smartsim._core.schemas import (
    DragonHandshakeRequest,
    DragonHandshakeResponse,
    DragonRequest,
    DragonResponse,
    DragonRunRequest,
    DragonRunResponse,
    DragonStopRequest,
    DragonStopResponse,
    DragonUpdateStatusRequest,
    DragonUpdateStatusResponse,
)
from smartsim.status import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_NEVER_STARTED,
    STATUS_RUNNING,
)


class DragonBackend:
    """The DragonBackend class is the main interface between
    SmartSim and Dragon. It is not intended to be user-facing,
    and will only be called by the Dragon entry-point script or
    by threads spawned by it.
    """

    def __init__(self) -> None:
        self._request_to_function: t.Mapping[
            str, t.Callable[[t.Any], DragonResponse]
        ] = {
            "run": self.run,
            "update_status": self.update_status,
            "stop": self.stop,
            "handshake": DragonBackend.handshake,
        }
        self.procs: t.Dict[str, Process] = {}

    def process_request(self, request: DragonRequest) -> DragonResponse:
        req_type = DragonRequest.parse_obj(request).request_type
        if not req_type:
            raise ValueError("Malformed request contains empty ``request_type`` field.")
        if req_type not in self._request_to_function:
            raise ValueError(f"Unknown request type {req_type}.")

        return self._request_to_function[req_type](request)

    def run(self, request: DragonRunRequest) -> DragonRunResponse:
        run_request = DragonRunRequest.parse_obj(request)

        exe = run_request.exe[0]
        if len(run_request.exe) > 1:
            exe_tail = run_request.exe[1:]
        else:
            exe_tail = []

        run_args = run_request.exe_args if run_request.exe_args is not None else []
        exe_args = tuple(exe_tail + run_args)

        proc = Process(
            target=exe,
            args=exe_args,
            cwd=run_request.path,
            env=run_request.env,
            stdout=Popen.PIPE,
            stderr=Popen.PIPE,
        )
        proc.start()
        self.procs[str(proc.puid)] = proc

        return DragonRunResponse(step_id=str(proc.puid))

    def update_status(
        self, request: DragonUpdateStatusRequest
    ) -> DragonUpdateStatusResponse:
        update_status_request = DragonUpdateStatusRequest.parse_obj(request)

        updated_statuses = {}
        for step_id in update_status_request.step_ids:
            if step_id in self.procs:
                proc = self.procs[step_id]
                if proc.is_alive:
                    updated_statuses[step_id] = (STATUS_RUNNING, None)
                else:
                    return_code = proc.returncode
                    status = STATUS_FAILED if return_code != 0 else STATUS_COMPLETED
                    updated_statuses[step_id] = (status, return_code)
            else:
                updated_statuses[step_id] = (STATUS_NEVER_STARTED, None)

        return DragonUpdateStatusResponse(statuses=updated_statuses)

    def stop(self, request: DragonStopRequest) -> DragonStopResponse:
        stop_request = DragonStopRequest.parse_obj(request)

        if stop_request.step_id in self.procs:
            # Technically we could just terminate, but what if
            # the application intercepts that and ignores it?
            self.procs[stop_request.step_id].kill()

        return DragonStopResponse()

    @staticmethod
    def handshake(request: DragonHandshakeRequest) -> DragonHandshakeResponse:
        DragonHandshakeRequest.parse_obj(request)

        return DragonHandshakeResponse()
