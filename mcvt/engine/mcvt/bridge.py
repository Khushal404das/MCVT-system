"""Connection from the Python engine to the PHP WebSocket hub.

The engine is a WebSocket *client* of the PHP server (role=engine). It pushes
JSON events and binary frame packets; the PHP hub fans them out to every
authenticated browser and relays operator commands (e.g. "follow VH-004")
back to the engine.

Binary frame packet: uint32 big-endian header length | JSON header | JPEG bytes
"""

from __future__ import annotations

import json
import logging
import struct
import threading
import time
from collections import OrderedDict, deque
from typing import Callable
from urllib.parse import urlencode

import websocket

log = logging.getLogger("bridge")


class Bridge:
    def __init__(self, url: str, token: str, on_command: Callable[[dict], None],
                 on_connect: Callable[[], None] | None = None):
        self.url = f"{url}?{urlencode({'role': 'engine', 'token': token})}"
        self.on_command = on_command
        self.on_connect = on_connect
        self.connected = False
        self._ws: websocket.WebSocket | None = None
        self._events: deque = deque(maxlen=2000)
        self._frames: OrderedDict[str, bytes] = OrderedDict()
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._warned = False

    # --------------------------------------------------------------- public
    def start(self) -> None:
        threading.Thread(target=self._connect_loop, name="bridge-conn", daemon=True).start()
        threading.Thread(target=self._send_loop, name="bridge-send", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass

    def send_event(self, payload: dict) -> None:
        if not self.connected:
            return
        with self._cv:
            self._events.append(json.dumps(payload, separators=(",", ":"), default=_json_default))
            self._cv.notify()

    def send_frame(self, camera_id: str, header: dict, jpeg: bytes) -> None:
        """Queue a frame; an unsent older frame of the same camera is replaced."""
        if not self.connected:
            return
        h = json.dumps(header, separators=(",", ":")).encode("utf-8")
        packet = struct.pack(">I", len(h)) + h + jpeg
        with self._cv:
            self._frames[camera_id] = packet
            self._frames.move_to_end(camera_id)
            self._cv.notify()

    # ------------------------------------------------------------- threads
    def _connect_loop(self) -> None:
        while not self._stop.is_set():
            try:
                ws = websocket.create_connection(self.url, timeout=10, enable_multithread=True)
                ws.settimeout(None)
                self._ws = ws
                self.connected = True
                self._warned = False
                log.info("Connected to the WebSocket hub")
                if self.on_connect:
                    self.on_connect()
                while not self._stop.is_set():
                    opcode, data = ws.recv_data()
                    if opcode == websocket.ABNF.OPCODE_CLOSE:
                        break
                    if opcode == websocket.ABNF.OPCODE_TEXT:
                        try:
                            self.on_command(json.loads(data.decode("utf-8")))
                        except Exception:
                            log.exception("Bad command from hub")
            except Exception as exc:
                if not self._warned:
                    log.warning("WebSocket hub not reachable (%s) - retrying every 3 s", exc)
                    self._warned = True
            finally:
                self.connected = False
                with self._cv:
                    self._frames.clear()
                    self._events.clear()
                if self._ws is not None:
                    try:
                        self._ws.close()
                    except Exception:
                        pass
                    self._ws = None
            self._stop.wait(3)

    def _send_loop(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                while not self._events and not self._frames and not self._stop.is_set():
                    self._cv.wait(0.5)
                if self._events:
                    item, binary = self._events.popleft(), False
                elif self._frames:
                    _, item = self._frames.popitem(last=False)
                    binary = True
                else:
                    continue
            ws = self._ws
            if ws is None or not self.connected:
                continue
            try:
                if binary:
                    ws.send_binary(item)
                else:
                    ws.send(item)
            except Exception as exc:
                log.debug("send failed: %s", exc)
                self.connected = False
                try:
                    ws.close()
                except Exception:
                    pass
                time.sleep(0.2)


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass
    return str(o)
