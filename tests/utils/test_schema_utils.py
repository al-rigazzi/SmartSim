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

import json

import pydantic
import pytest

from smartsim._core.schemas.utils import SchemaRegistry, _Message

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b


class Person(pydantic.BaseModel):
    name: str
    age: int


class Book(pydantic.BaseModel):
    title: str
    num_pages: int


def test_schema_registrartion():
    registry = SchemaRegistry()
    assert registry._map == {}

    registry.register("person")(Person)
    assert registry._map == {"person": Person}

    registry.register("book")(Book)
    assert registry._map == {"person": Person, "book": Book}


def test_cannot_register_a_schema_under_an_empty_str():
    registry = SchemaRegistry()
    with pytest.raises(KeyError, match="Key cannot be the empty string"):
        registry.register("")


@pytest.mark.parametrize(
    "delim",
    (
        pytest.param(SchemaRegistry._DEFAULT_DELIMITER, id="default delimiter"),
        pytest.param("::", id="custom delimiter"),
    ),
)
def test_schema_to_string(delim):
    registry = SchemaRegistry(delim)
    registry.register("person")(Person)
    registry.register("book")(Book)
    person = Person(name="Bob", age=36)
    book = Book(title="The Greatest Story of All Time", num_pages=10_000)
    assert registry.to_string(person) == str(
        _Message(person, "person", registry._msg_delim)
    )
    assert registry.to_string(book) == str(_Message(book, "book", registry._msg_delim))


def test_registry_errors_if_types_overloaded():
    registry = SchemaRegistry()
    registry.register("schema")(Person)

    with pytest.raises(KeyError):
        registry.register("schema")(Book)


def test_registry_errors_if_msg_delim_is_empty():
    with pytest.raises(ValueError, match="empty string"):
        SchemaRegistry("")


def test_registry_errors_if_msg_type_registered_with_delim_present():
    registry = SchemaRegistry("::")
    with pytest.raises(ValueError, match="cannot contain delimiter"):
        registry.register("new::type")


def test_registry_errors_on_unknown_schema():
    registry = SchemaRegistry()
    registry.register("person")(Person)

    with pytest.raises(TypeError):
        registry.to_string(Book(title="The Shortest Story of All Time", num_pages=1))


@pytest.mark.parametrize(
    "delim",
    (
        pytest.param(SchemaRegistry._DEFAULT_DELIMITER, id="default delimiter"),
        pytest.param("::", id="custom delimiter"),
    ),
)
def test_registry_correctly_maps_to_expected_type(delim):
    registry = SchemaRegistry(delim)
    registry.register("person")(Person)
    registry.register("book")(Book)
    person = Person(name="Bob", age=36)
    book = Book(title="The Most Average Story of All Time", num_pages=500)
    assert (
        registry.from_string(str(_Message(person, "person", registry._msg_delim)))
        == person
    )
    assert (
        registry.from_string(str(_Message(book, "book", registry._msg_delim))) == book
    )


def test_registery_errors_if_type_key_not_recognized():
    registry = SchemaRegistry()
    registry.register("person")(Person)

    with pytest.raises(ValueError, match="^No type of value .* registered$"):
        registry.from_string(
            str(_Message(Person(name="Grunk", age=5_000), "alien", registry._msg_delim))
        )


def test_registry_errors_if_type_key_is_missing():
    registry = SchemaRegistry("::")
    registry.register("person")(Person)

    with pytest.raises(ValueError, match="Failed to determine schema type"):
        registry.from_string("This string does not contain a delimiter")
