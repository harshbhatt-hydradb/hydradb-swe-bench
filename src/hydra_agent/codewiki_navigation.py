"""Documentation navigation with explicit failures and verifiable content reads."""

import json


def documentation_paths(docs, path=()):
    pages = []
    if not isinstance(docs, dict):
        return pages
    if readable_content(docs.get("content")):
        pages.append({"title": docs.get("title", ""), "path": [*path, "content"]})
    for index, page in enumerate(docs.get("subpages", [])):
        pages.extend(documentation_paths(page, (*path, "subpages", index)))
    return pages


def normalize_path(path):
    if not isinstance(path, list):
        raise TypeError("Each path must be a list of keys and indices")
    # An extra singleton wrapper has only one possible interpretation. Do not
    # flatten multiple nested paths or coerce malformed keys into strings.
    for _ in range(8):
        if len(path) == 1 and isinstance(path[0], list):
            path = path[0]
        else:
            break
    if any(type(key) not in (str, int) or (type(key) is int and key < 0) for key in path):
        raise ValueError("Path keys must be strings or nonnegative integers")
    return path


def resolve_path(docs, path):
    path = normalize_path(path)
    value = docs
    if path and isinstance(docs, dict) and path[0] == docs.get("title"):
        path = path[1:]
    for key in path:
        if isinstance(value, dict):
            if key in value:
                value = value[key]
                continue
            children = value.get("subpages", [])
            if type(key) is int:
                value = children[key]
            else:
                matches = [page for page in children if page.get("title") == key]
                if len(matches) > 1:
                    raise ValueError("Ambiguous page title; use a JSON key/index path")
                if not matches:
                    raise KeyError(key)
                value = matches[0]
        elif isinstance(value, list):
            index = int(key)
            if index < 0:
                raise ValueError("Negative index")
            value = value[index]
        else:
            raise KeyError(key)
    return value


def readable_content(value):
    """Recognize body text, excluding empty nodes and tree placeholders."""
    if isinstance(value, str):
        return bool(value.strip()) and value.strip() != "<detail_content>"
    if isinstance(value, dict):
        return any(readable_content(item) for item in value.values())
    if isinstance(value, list):
        return any(readable_content(item) for item in value)
    return False


def contains_documentation(value, path=()):
    if "content" in path:
        return readable_content(value)
    if isinstance(value, dict):
        return readable_content(value.get("content")) or any(
            contains_documentation(page) for page in value.get("subpages", [])
        )
    if isinstance(value, list):
        return any(contains_documentation(page) for page in value)
    # Reading a title/description is navigation, not a read of the page body.
    return False


def navigation_succeeded(results):
    if not isinstance(results, list) or not results:
        return False
    for entry in results:
        if not isinstance(entry, dict) or "error" in entry or "content" not in entry:
            return False
        try:
            path = normalize_path(entry.get("resolved_path", entry.get("path", [])))
        except (TypeError, ValueError):
            return False
        if not contains_documentation(entry["content"], path):
            return False
    return True


def navigate(docs, paths, *, max_bytes=100_000):
    if not isinstance(paths, list) or not 1 <= len(paths) <= 20:
        raise ValueError("Supply between one and twenty paths")
    results = []
    for path in paths:
        try:
            resolved = normalize_path(path)
            value = resolve_path(docs, resolved)
            if resolved and resolved[-1] in ("title", "description"):
                page = resolve_path(docs, resolved[:-1])
                if isinstance(page, dict) and readable_content(page.get("content")):
                    value = {"requested_metadata": value, "content": page["content"]}
                    resolved = [*resolved[:-1], "content"]
            if not contains_documentation(value, resolved):
                raise ValueError("Path has no readable page content; read a listed content path")
            entry = {"path": path, "content": value}
            if path != resolved:
                entry["resolved_path"] = resolved
            results.append(entry)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            results.append(
                {
                    "path": path,
                    "error": str(exc)
                    if isinstance(exc, ValueError)
                    else "Unknown documentation path",
                    "available_paths": documentation_paths(docs),
                }
            )
    if max_bytes is not None and len(json.dumps(results).encode()) > max_bytes:
        return {
            "error": "Requested sections exceed tool budget; request fewer or narrower paths",
            "available_paths": documentation_paths(docs),
        }
    return results
