import typing as t

import pydantic
import pydantic.dataclasses

_SchemaT = t.TypeVar("_SchemaT", bound=pydantic.BaseModel)


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
