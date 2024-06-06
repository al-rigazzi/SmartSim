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

import time
from abc import ABC, abstractmethod

from smartsim.log import get_logger

logger = get_logger(__name__)


class ServiceHost(ABC):
    """Base contract for standalone entrypoint scripts. Defines API for entrypoint
    behaviors (event loop, automatic shutdown, cooldown) as well as simple
    hooks for status changes"""

    def __init__(
        self, as_service: bool = False, cooldown: int = 0, loop_delay: int = 0
    ) -> None:
        """Initialize the ServiceHost
        :param as_service: Determines if the host will run until shutdown criteria
        are met or as a run-once instance
        :param cooldown: Period of time to allow service to run before automatic
        shutdown, in seconds. A non-zero, positive integer."""
        self._as_service = as_service
        """If the service should run until shutdown function returns True"""
        self._cooldown = cooldown
        """Duration of a cooldown period between requests to the service
        before shutdown"""
        self._loop_delay = loop_delay
        """Forced delay between iterations of the event loop"""

    @abstractmethod
    def _on_iteration(self, timestamp: int) -> None:
        """The user-defined event handler. Executed repeatedly until shutdown
        conditions are satisfied and cooldown is elapsed.
        :param timestamp: the timestamp at the start of the event loop iteration
        """

    @abstractmethod
    def _can_shutdown(self) -> bool:
        """Return true when the criteria to shut down the service are met."""

    def _on_start(self) -> None:
        """Empty hook method for use by subclasses. Called on initial entry into
        ServiceHost `execute` event loop before `_on_iteration` is invoked."""
        logger.debug(f"Starting {self.__class__.__name__}")

    def _on_shutdown(self) -> None:
        """Empty hook method for use by subclasses. Called immediately after exiting
        the main event loop during automatic shutdown."""
        logger.debug(f"Shutting down {self.__class__.__name__}")

    def _on_cooldown(self) -> None:
        """Empty hook method for use by subclasses. Called on every event loop
        iteration immediately upon exceeding the cooldown period"""
        logger.debug(f"Cooldown exceeded by {self.__class__.__name__}")

    def execute(self) -> None:
        """The main event loop of a service host. Evaluates shutdown criteria and
        combines with a cooldown period to allow automatic service termination.
        Responsible for executing calls to subclass implementation of `_on_iteration`"""
        self._on_start()

        start_ts = time.time_ns()
        last_ts = start_ts
        running = True
        elapsed_cooldown = 0
        nanosecond_scale_factor = 1000000000
        cooldown_ns = self._cooldown * nanosecond_scale_factor

        # if we're run-once, use cooldown to short circuit
        if not self._as_service:
            self._cooldown = 1
            last_ts = start_ts - (cooldown_ns * 2)

        while running:
            self._on_iteration(start_ts)

            eligible_to_quit = self._can_shutdown()

            if self._cooldown and not eligible_to_quit:
                # reset timer any time cooldown is interrupted
                elapsed_cooldown = 0

            # allow service to shutdown if no cooldown period applies...
            running = not eligible_to_quit

            # ... but verify we don't have remaining cooldown time
            if self._cooldown:
                elapsed_cooldown += start_ts - last_ts
                remaining = cooldown_ns - elapsed_cooldown
                running = remaining > 0

                rem_in_s = remaining / nanosecond_scale_factor

                if not running:
                    cd_in_s = cooldown_ns / nanosecond_scale_factor
                    logger.info(f"cooldown {cd_in_s}s exceeded by {abs(rem_in_s):.2f}s")
                    self._on_cooldown()
                    continue

                logger.debug(f"cooldown remaining {abs(rem_in_s):.2f}s")

            last_ts = start_ts
            start_ts = time.time_ns()

            if self._loop_delay:
                time.sleep(abs(self._loop_delay))

        self._on_shutdown()
