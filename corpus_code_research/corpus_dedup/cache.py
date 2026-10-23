from __future__ import annotations

import fcntl
import json
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def file_lock(path: Path, *, timeout_s: float = 300.0, poll_interval_s: float = 0.2) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o664)
    start = time.monotonic()
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() - start > timeout_s:
                os.close(fd)
                raise TimeoutError(f"Timed out waiting for lock: {path}")
            time.sleep(poll_interval_s)
    try:
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "acquired_at_unix": time.time(),
            }
        ).encode("utf8")
        os.ftruncate(fd, 0)
        os.write(fd, payload)
        os.fsync(fd)
        yield
    finally:
        try:
            os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def atomic_replace(temp_path: Path, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.replace(final_path)
