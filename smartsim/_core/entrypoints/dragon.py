# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterpris
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

import argparse
import json
import os
import signal
import socket
import sys
import textwrap
import typing as t
from types import FrameType

import urllib3
import urllib3.util
import zmq
import zmq.auth.thread

from smartsim._core.launcher.dragon import dragonSockets
from smartsim._core.launcher.dragon.dragonBackend import DragonBackend
from smartsim._core.schemas import DragonBootstrapRequest, DragonBootstrapResponse
from smartsim._core.utils.network import find_free_port, get_best_interface_and_address
from smartsim.log import get_logger

logger = get_logger("Dragon Server")

# kill is not catchable
SIGNALS = [signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGABRT]

SHUTDOWN_INITIATED = False


def handle_signal(signo: int, _frame: t.Optional[FrameType]) -> None:
    if not signo:
        logger.info("Received signal with no signo")
    else:
        logger.info(f"Received {signo}")
    cleanup()


"""
Dragon server entrypoint script
"""


def print_summary(network_interface: str, ip_address: str) -> None:
    zmq_config = {"interface": network_interface, "address": ip_address}

    with open("dragon_config.log", "w", encoding="utf-8") as dragon_config_log:
        dragon_config_log.write(
            textwrap.dedent(f"""\
                -------- Dragon Configuration --------
                IPADDRESS: {ip_address}
                NETWORK: {network_interface}
                HOSTNAME: {socket.gethostname()}
                DRAGON_SERVER_CONFIG: {json.dumps(zmq_config)}
                --------------------------------------

                --------------- Output ---------------

                """),
        )


def configure_zmq_context(zmq_context: zmq.Context[t.Any]) -> None:
    zmq_context.setsockopt(zmq.SNDTIMEO, value=1000)
    zmq_context.setsockopt(zmq.RCVTIMEO, value=1000)
    zmq_context.setsockopt(zmq.REQ_CORRELATE, 1)
    zmq_context.setsockopt(zmq.REQ_RELAXED, 1)


def allocate_head_socket(
    zmq_context: zmq.Context[t.Any], dragon_head_address: str
) -> zmq.Socket[t.Any]:
    """Create a socket to be used for receiving messages from a DragonLauncher"""
    logger.debug(f"Opening head socket at {dragon_head_address}")

    dragon_head_socket = dragonSockets.get_secure_socket(zmq_context, zmq.REP, True)
    dragon_head_socket.bind(dragon_head_address)

    logger.info(f"Head socket listening at {dragon_head_address}")
    return dragon_head_socket


def allocate_launcher_socket(
    zmq_context: zmq.Context[t.Any], launcher_address: str
) -> zmq.Socket[t.Any]:
    """Create a socket to be used for receiving messages from a DragonLauncher"""
    logger.debug(f"Opening callback socket at {launcher_address}")

    launcher_socket = dragonSockets.get_secure_socket(zmq_context, zmq.REQ, False)
    launcher_socket.connect(launcher_address)

    logger.info(f"Callback socket listening at {launcher_address}")
    return launcher_socket


def transform_address(launching_address: str, dragon_head_address: str) -> str:
    """Compare the launch and dragon head addresses to determine if
    traffic can be routed through localhost."""
    launch_url = urllib3.util.parse_url(launching_address)
    head_url = urllib3.util.parse_url(dragon_head_address)

    # compare the launch url w/the initial address; convert
    # to localhost to reduce network traffic
    if launch_url.hostname == head_url.hostname and head_url.hostname is not None:
        return head_url.url.replace(head_url.hostname, "localhost")

    return head_url.url


def run(
    dragon_head_address: str,
    dragon_pid: int,
    dragon_head_socket: zmq.Socket[t.Any],
) -> None:
    dragon_backend = DragonBackend(pid=dragon_pid)
    server = dragonSockets.as_server(dragon_head_socket)

    while not (dragon_backend.should_shutdown or SHUTDOWN_INITIATED):
        logger.debug(f"Listening to {dragon_head_address}")

        try:
            req = server.recv()
            logger.debug(f"Received {type(req).__name__} {req}")
        except zmq.Again:
            dragon_backend.update()
            continue

        resp = dragon_backend.process_request(req)

        try:
            logger.debug(f"Sending {type(resp).__name__} {resp}")
            server.send(resp)
        except zmq.Again:
            logger.error("Could not send response back to launcher.")

        dragon_backend.print_status()
        dragon_backend.update()

        if dragon_backend.should_shutdown or SHUTDOWN_INITIATED:
            logger.info("Shutdown has been requested")


def main(args: argparse.Namespace, zmq_context: zmq.Context[t.Any]) -> int:
    if_config = get_best_interface_and_address()
    interface = if_config.interface
    address = if_config.address
    if not interface:
        raise ValueError("Net interface could not be determined")

    if not args.launching_address:
        logger.warning("Launching address was not supplied to the dragon entrypoint")
        return 0

    configure_zmq_context(zmq_context)

    port = find_free_port(start=6995)
    dragon_head_address = transform_address(
        args.launching_address, f"tcp://{address}:{port}"
    )

    # set up an authenticator to handle authentication duties
    authenticator = dragonSockets.get_authenticator(zmq_context)

    # grab the socket we want before sending back the address (to ensure it works)
    dragon_head_socket = allocate_head_socket(context, dragon_head_address)

    # connect to the launcher and let it know where dragon head is exposed
    launcher_socket = allocate_launcher_socket(context, args.launching_address)
    client = dragonSockets.as_client(launcher_socket)

    # send message to the launcher with dragon head socket address
    client.send(DragonBootstrapRequest(address=dragon_head_address))
    # ... and expect an immediate callback confirming they received it
    response = client.recv()
    if not isinstance(response, DragonBootstrapResponse):
        raise ValueError(
            "Could not receive connection confirmation from launcher. Aborting."
        )

    try:
        print_summary(interface, dragon_head_address)
        run(
            dragon_head_address=dragon_head_address,
            dragon_pid=response.dragon_pid,
            dragon_head_socket=dragon_head_socket,
        )
    except Exception as e:
        logger.error(f"Dragon server failed with {e}", exc_info=True)
        return os.EX_SOFTWARE
    finally:
        if authenticator.is_alive():
            authenticator.stop()

    logger.info("Shutting down! Bye bye!")
    return 0


def cleanup() -> None:
    global SHUTDOWN_INITIATED  # pylint: disable=global-statement
    logger.debug("Cleaning up")
    SHUTDOWN_INITIATED = True


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    logger.info("Dragon server started")

    parser = argparse.ArgumentParser(
        prefix_chars="+", description="SmartSim Dragon Head Process"
    )
    parser.add_argument(
        "+launching_address",
        type=str,
        help="Address of launching process if a ZMQ connection can be established",
        required=False,
    )
    parser.add_argument(
        "+interface", type=str, help="Network Interface name", required=False
    )
    args_ = parser.parse_args()

    # make sure to register the cleanup before the start
    # the process so our signaller will be able to stop
    # the database process.
    for sig in SIGNALS:
        signal.signal(sig, handle_signal)

    context = zmq.Context()

    sys.exit(main(args_, context))
