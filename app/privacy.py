"""Contexte de confidentialité, isolé par tour/thread et transmis aux outils."""
from contextlib import contextmanager
from contextvars import ContextVar

_private = ContextVar("kira_private", default=False)


def is_private():
    return _private.get()


def mark_private():
    _private.set(True)


@contextmanager
def scope(private=False):
    token = _private.set(bool(private))
    try:
        yield
    finally:
        _private.reset(token)
