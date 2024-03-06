from pathlib import Path, PurePath
from typing import Any, Callable
import json


def json_default(obj: Any):
    if hasattr(obj, '__iter__'):
        return list(obj)
    else:
        return str(obj)


def dump_json(obj: Any, path: Path | str, default: Callable = json_default) -> None:
    if not isinstance(path, PurePath):
        path = Path(path)
    try:
        with path.open('w', encoding='utf-8') as f:
            json.dump(obj, f, default=default)
    except UnicodeDecodeError:
        return


def load_json(path: Path | str) -> dict[str, Any]:
    if not isinstance(path, PurePath):
        path = Path(path)
    try:
        with path.open(encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        return {}
