import itertools

import pytest

from smartsim._core.utils import helpers


@pytest.mark.parametrize(
    "func_1, func_2, func_3",
    itertools.permutations((lambda x: x + 3, lambda x: x * 2, lambda x: x // 5)),
)
def test_pipline(func_1, func_2, func_3):
    x = 30
    assert func_3(func_2(func_1(x))) == (
        helpers.Pipeline(x).pipe(func_1).pipe(func_2).pipe(func_3).result()
    )
