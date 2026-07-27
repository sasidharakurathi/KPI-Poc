"""
Local TCP proxy that works around an FFmpeg RTSP-client limitation: when a
server offers multiple WWW-Authenticate Digest challenges in one 401
response (e.g. both algorithm="MD5" and algorithm="SHA-256"), FFmpeg's RTSP
client fails to pick the one it actually supports (MD5) and never retries
with credentials at all -- producing a permanent 401 even though the
credentials are correct and MD5 is a valid option.

This proxy forwards a single RTSP TCP connection to the real camera nearly
byte-for-byte, except it rewrites 401 responses to drop any non-MD5
WWW-Authenticate line before relaying them. FFmpeg then sees only the MD5
challenge and authenticates normally -- no digest computation is done here;
the client (FFmpeg) still computes its own response, same as always.
"""
import logging
import re
import socket
import threading
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

_BUF = 65536
_MAX_HEADER_WAIT = 1 << 20


def localize_url(original_url: str, local_host: str, local_port: int) -> str:
    """Rewrite only the host:port of an rtsp:// URL, preserving userinfo/path/query."""
    parts = urlsplit(original_url)
    netloc = parts.netloc
    if "@" in netloc:
        userinfo, _, _hostport = netloc.rpartition("@")
        new_netloc = f"{userinfo}@{local_host}:{local_port}"
    else:
        new_netloc = f"{local_host}:{local_port}"
    return urlunsplit((parts.scheme, new_netloc, parts.path, parts.query, parts.fragment))


def _keep_only_md5_challenge(header_block: bytes) -> bytes:
    lines = header_block.split(b"\r\n")
    kept = []
    for line in lines:
        if line.lower().startswith(b"www-authenticate:") and b"sha-256" in line.lower():
            continue
        kept.append(line)
    return b"\r\n".join(kept)


class RtspAuthFixProxy:
    """Listens on a local port and relays one upstream RTSP connection at a
    time (per incoming client), fixing up 401 responses along the way."""

    def __init__(self, upstream_host: str, upstream_port: int, local_port: int = 0) -> None:
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", local_port))
        self._listener.listen(5)
        self.local_port = self._listener.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="rtsp-auth-proxy"
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._listener.close()
        except OSError:
            pass

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                client_sock, _ = self._listener.accept()
            except OSError:
                break
            threading.Thread(
                target=self._handle_client, args=(client_sock,), daemon=True
            ).start()

    def _handle_client(self, client_sock: socket.socket) -> None:
        try:
            upstream_sock = socket.create_connection(
                (self.upstream_host, self.upstream_port), timeout=10
            )
        except OSError:
            logger.exception("[rtsp-auth-proxy] failed to connect upstream")
            client_sock.close()
            return

        t1 = threading.Thread(
            target=self._relay_passthrough, args=(client_sock, upstream_sock), daemon=True
        )
        t2 = threading.Thread(
            target=self._relay_fix_auth, args=(upstream_sock, client_sock), daemon=True
        )
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        for s in (client_sock, upstream_sock):
            try:
                s.close()
            except OSError:
                pass

    def _relay_passthrough(self, src: socket.socket, dst: socket.socket) -> None:
        """App -> camera: forwarded unchanged, byte for byte."""
        try:
            while True:
                data = src.recv(_BUF)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        finally:
            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    def _relay_fix_auth(self, src: socket.socket, dst: socket.socket) -> None:
        """Camera -> app: passthrough, except 401 responses get their
        WWW-Authenticate challenges filtered down to MD5 only. Binary
        interleaved RTP/RTCP frames ('$' + channel + 2-byte length) are
        detected and passed through without being parsed as text."""
        buf = b""
        try:
            while True:
                chunk = src.recv(_BUF)
                if not chunk:
                    break
                buf += chunk

                while buf:
                    if buf[0:1] == b"$":
                        if len(buf) < 4:
                            break
                        frame_len = int.from_bytes(buf[2:4], "big")
                        total = 4 + frame_len
                        if len(buf) < total:
                            break
                        dst.sendall(buf[:total])
                        buf = buf[total:]
                        continue

                    header_end = buf.find(b"\r\n\r\n")
                    if header_end == -1:
                        if len(buf) > _MAX_HEADER_WAIT:
                            dst.sendall(buf)
                            buf = b""
                        break

                    header_block = buf[:header_end]
                    body_start = header_end + 4

                    content_length = 0
                    m = re.search(rb"content-length:\s*(\d+)", header_block, re.IGNORECASE)
                    if m:
                        content_length = int(m.group(1))

                    if len(buf) < body_start + content_length:
                        break   # wait for the rest of the body to arrive

                    body = buf[body_start:body_start + content_length]
                    buf = buf[body_start + content_length:]

                    status_line = header_block.split(b"\r\n", 1)[0]
                    if b" 401 " in status_line:
                        header_block = _keep_only_md5_challenge(header_block)

                    dst.sendall(header_block + b"\r\n\r\n" + body)
        except OSError:
            pass
        finally:
            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass
