from __future__ import unicode_literals

import hashlib
from collections.abc import Iterable
from typing import Any


def _recursive_repr(item: str | list[Any] | dict[str, Any]) -> str:
    """Hack around python `repr` to deterministically represent dictionaries.

    This is able to represent more things than json.dumps, since it does not require
    things to be JSON serializable (e.g. datetimes).
    """
    if isinstance(item, str):
        result = str(item)
    elif isinstance(item, list):
        result = '[{}]'.format(', '.join([_recursive_repr(x) for x in item]))
    elif isinstance(item, dict):
        kv_pairs = ['{}: {}'.format(_recursive_repr(k), _recursive_repr(item[k])) for k in sorted(item)]
        result = '{' + ', '.join(kv_pairs) + '}'
    else:
        result = repr(item)
    return result


def get_hash(item: str | list[Any] | dict[str, Any]) -> str:
    repr_ = _recursive_repr(item).encode('utf-8')
    return hashlib.md5(repr_).hexdigest()


def get_hash_int(item: str | list[Any] | dict[str, Any]) -> int:
    return int(get_hash(item), base=16)


def escape_chars(text: str | int | tuple[int, int], chars: str) -> str:
    """Helper function to escape uncomfortable characters."""
    text = str(text)
    _chars = list(set(chars))
    if '\\' in _chars:
        _chars.remove('\\')
        _chars.insert(0, '\\')
    for ch in _chars:
        text = text.replace(ch, '\\' + ch)
    return text


def convert_kwargs_to_cmd_line_args(kwargs: dict[str, Any]) -> list[str]:
    """Helper function to build command line arguments out of dict."""
    args = []
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if isinstance(v, Iterable) and not isinstance(v, str):
            for value in v:
                args.append('-{}'.format(k))
                if value is not None:
                    args.append('{}'.format(value))
            continue
        args.append('-{}'.format(k))
        if v is not None:
            args.append('{}'.format(v))
    return args
