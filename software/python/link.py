import itertools
import json
import socket

import msgpack

SOCKET_PATH = "/var/run/arduino-router.sock"
TIMEOUT = 2.0

REQUEST = 0
RESPONSE = 1

_ids = itertools.count()
_sock = None
_unpacker = None


def connect(path=SOCKET_PATH):
    global _sock, _unpacker
    _sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    _sock.settimeout(TIMEOUT)
    _sock.connect(path)
    _unpacker = msgpack.Unpacker(raw=False)


def disconnect():
    global _sock
    if _sock is not None:
        _sock.close()
    _sock = None


def wait_response(msgid):
    while True:
        for msg in _unpacker:
            if msg[0] == RESPONSE and msg[1] == msgid:
                return msg
        data = _sock.recv(4096)
        if not data:
            raise ConnectionError("router closed the connection")
        _unpacker.feed(data)


def call(method, *args):
    if _sock is None:
        connect()
    msgid = next(_ids) & 0xFFFFFFFF
    try:
        _sock.sendall(msgpack.packb([REQUEST, msgid, method, list(args)]))
        _, _, error, result = wait_response(msgid)
    except (OSError, ValueError):
        disconnect()
        raise
    if error is not None:
        raise RuntimeError(f"{method}: {error}")
    return result


def parse_edge(edge):
    a, b = edge[0], edge[1]
    wired = int(edge[2]) if len(edge) > 2 else 1
    return a, b, wired


def parse_mesh(raw):
    mesh = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    current = mesh.get("current")
    return {
        "nodes": {n: float(v) for n, v in mesh["nodes"].items()},
        "temp": float(mesh.get("temp", 0.0)),
        "current": float(current) if current is not None else None,
        "edges": [parse_edge(e) for e in mesh.get("edges", [])],
        "layout": {n: (float(p[0]), float(p[1])) for n, p in mesh.get("layout", {}).items()},
    }


def get_mesh():
    return parse_mesh(call("get_mesh"))
