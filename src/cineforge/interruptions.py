"""Retain cancellation across privacy handlers until their owned cleanup finishes.

Each decorated boundary owns a separate flag. Handlers remember only a boolean,
never an exception/message/traceback. Nested boundaries rethrow a fresh interrupt;
cleanup loops may finish their existing bounded attempts before it propagates.
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")
_interrupt_state: ContextVar[list[bool] | None] = ContextVar("interrupt_state", default=None)


def remember_interruption(error: BaseException) -> None:
    if isinstance(error, KeyboardInterrupt):
        state = _interrupt_state.get()
        if state is None:
            raise KeyboardInterrupt from None
        state[0] = True


def preserve_interruption(function: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(function)
    def call(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        state = [False]
        token = _interrupt_state.set(state)
        try:
            try:
                result = function(*args, **kwargs)
            except KeyboardInterrupt:
                state[0] = True
            except BaseException:
                if not state[0]:
                    raise
        finally:
            _interrupt_state.reset(token)
        # Leave every handler before raising: no private exception context remains.
        if state[0]:
            raise KeyboardInterrupt from None
        return result
    return call
