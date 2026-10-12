"""Minimal YAML-subset parser (fallback when PyYAML is absent).

Supports the small subset used by config files:
  key: value        (scalars / inline lists like [8, 16, 32])
  key:              (nested mapping)
    sub: value
  key:              (block list)
    - item
    - key: value    (list of mappings)
  # comments, blank lines, and inline comments after a scalar value.

Returns a nested dict/list structure. Only needs to be good enough for our
own config format; it is not a general YAML parser.
"""
from typing import Any, Dict, List


def _strip_comment(line: str) -> str:
    # remove a ` #` comment only when preceded by whitespace (keeps values
    # that legitimately contain '#'); also handle a leading full-line comment
    for i, ch in enumerate(line):
        if ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i].rstrip()
    return line.rstrip()


def _scalar(text: str) -> Any:
    t = text.strip()
    if t == "":
        return None
    if t.startswith("[") and t.endswith("]"):
        inner = t[1:-1].strip()
        if inner == "":
            return []
        return [_scalar(x) for x in inner.split(",")]
    low = t.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", "~"):
        return None
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    if len(t) >= 2 and t[0] in ("'", '"') and t[-1] == t[0]:
        return t[1:-1]
    return t


def parse(text: str) -> Dict[str, Any]:
    lines = []
    for raw in text.splitlines():
        s = _strip_comment(raw)
        if s.strip() == "":
            continue
        indent = len(s) - len(s.lstrip(" "))
        lines.append((indent, s.strip()))

    root: Dict[str, Any] = {}
    # stack of (indent, container, parent, key):
    #   container = the dict/list to fill now
    #   parent    = the dict/list that holds `container` (used to swap a dict
    #               placeholder for a list when a block list starts)
    #   key       = the key under which container lives in parent (dict case)
    stack: List[tuple] = [(-1, root, None, None)]

    for indent, content in lines:
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent_indent, container, parent, key = stack[-1]

        if content.startswith("- "):
            item = content[2:].strip()
            if isinstance(container, dict):
                # a block list started under a key that was pre-created as a
                # dict placeholder -> replace it with a list in the parent
                lst: Any = []
                if parent is not None and key is not None:
                    parent[key] = lst
                container = lst
                stack[-1] = (parent_indent, lst, parent, key)
            if ":" in item and not item.startswith(("http:", "https:")):
                k, v = item.split(":", 1)
                k = k.strip()
                d: Any = {}
                container.append(d)
                if v.strip():
                    d[k] = _scalar(v.strip())
                stack.append((indent, d, container, k))
            else:
                container.append(_scalar(item))
        elif ":" in content:
            k, v = content.split(":", 1)
            k = k.strip()
            v = v.strip()
            if isinstance(container, list):
                d: Any = {}
                container.append(d)
                if v:
                    d[k] = _scalar(v)
                stack.append((indent, d, container, k))
            else:
                if v == "":
                    child: Any = {}
                    container[k] = child
                    stack.append((indent, child, container, k))
                else:
                    container[k] = _scalar(v)
        else:
            if isinstance(container, list):
                container.append(_scalar(content))

    return root
