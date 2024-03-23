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

import dataclasses
import typing as t

import pydantic
import pydantic.dataclasses

if t.TYPE_CHECKING:
    from zmq.sugar.socket import Socket

_SchemaT = t.TypeVar("_SchemaT", bound=pydantic.BaseModel)
_SendT = t.TypeVar("_SendT", bound=pydantic.BaseModel)
_RecvT = t.TypeVar("_RecvT", bound=pydantic.BaseModel)


@t.final
@pydantic.dataclasses.dataclass(frozen=True)
class _Message(t.Generic[_SchemaT]):
    payload: _SchemaT
    header: str = pydantic.Field(min_length=1)
    delimiter: str = pydantic.Field(min_length=1)

    def __str__(self) -> str:
        return self.delimiter.join((self.header, self.payload.json()))

    @classmethod
    def from_str(
        cls, str_: str, delimiter: str, payload_type: t.Type[_SchemaT]
    ) -> "_Message[_SchemaT]":
        header, payload = str_.split(delimiter, 1)
        return cls(payload_type.parse_raw(payload), header, delimiter)


class SchemaRegistry(t.Generic[_SchemaT]):
    _DEFAULT_DELIMITER = "|"

    def __init__(
        self,
        message_delimiter: str = _DEFAULT_DELIMITER,
        init_map: t.Optional[t.Mapping[str, t.Type[_SchemaT]]] = None,
    ):
        if not message_delimiter:
            raise ValueError("Message delimiter cannot be an empty string")
        self._msg_delim = message_delimiter
        self._map = dict(init_map) if init_map else {}

    def register(self, key: str) -> t.Callable[[t.Type[_SchemaT]], t.Type[_SchemaT]]:
        if self._msg_delim in key:
            _msg = f"Registry key cannot contain delimiter `{self._msg_delim}`"
            raise ValueError(_msg)
        if not key:
            raise KeyError("Key cannot be the empty string")
        if key in self._map:
            raise KeyError(f"Key `{key}` has already been registered for this parser")

        def _register(cls: t.Type[_SchemaT]) -> t.Type[_SchemaT]:
            self._map[key] = cls
            return cls

        return _register

    def to_string(self, schema: _SchemaT) -> str:
        return str(self._to_message(schema))

    def _to_message(self, schema: _SchemaT) -> _Message[_SchemaT]:
        reverse_map = dict((v, k) for k, v in self._map.items())
        try:
            val = reverse_map[type(schema)]
        except KeyError:
            raise TypeError(f"Unregistered schema type: {type(schema)}") from None
        return _Message(schema, val, self._msg_delim)

    def from_string(self, str_: str) -> _SchemaT:
        try:
            type_, _ = str_.split(self._msg_delim, 1)
        except ValueError:
            _msg = f"Failed to determine schema type of the string {repr(str_)}"
            raise ValueError(_msg) from None
        try:
            cls = self._map[type_]
        except KeyError:
            raise ValueError(f"No type of value `{type_}` is registered") from None
        msg = _Message.from_str(str_, self._msg_delim, cls)
        return self._from_message(msg)

    @staticmethod
    def _from_message(msg: _Message[_SchemaT]) -> _SchemaT:
        return msg.payload


@dataclasses.dataclass(frozen=True)
class SocketSchemaTranslator(t.Generic[_SendT, _RecvT]):
    socket: "Socket[t.Any]"
    _send_registry: SchemaRegistry[_SendT]
    _recv_registry: SchemaRegistry[_RecvT]

    def send(self, schema: _SendT, flags: int = 0) -> None:
        self.socket.send_string(self._send_registry.to_string(schema), flags)

    def recv(self) -> _RecvT:
        return self._recv_registry.from_string(self.socket.recv_string())
