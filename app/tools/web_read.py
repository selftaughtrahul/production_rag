"""Bounded public-web page reader with SSRF protections."""

from __future__ import annotations

import asyncio
import http.client
import ipaddress
import socket
import ssl
from typing import Type
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from pydantic import BaseModel, Field

from app.tools.base import BaseAgentTool

_MAX_BYTES = 1_000_000
_MAX_REDIRECTS = 3
_BLOCKED_HOSTS = frozenset(
    {"metadata.google.internal", "metadata.aws.internal"}
)
_NAT64_WELL_KNOWN = ipaddress.ip_network("64:ff9b::/96")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and data.strip():
            self.parts.append(data.strip())


def _extract_text(raw: str) -> str:
    parser = _TextExtractor()
    parser.feed(raw)
    return "\n".join(parser.parts)


def _resolve_public_url(url: str) -> tuple[str, str, int, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only public HTTP(S) URLs are allowed.")
    if parsed.username or parsed.password or parsed.hostname.lower() in _BLOCKED_HOSTS:
        raise ValueError("URL credentials and metadata hosts are not allowed.")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = socket.getaddrinfo(parsed.hostname, port)
    except socket.gaierror as exc:
        raise ValueError("URL hostname could not be resolved.") from exc
    public_addresses: list[str] = []
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("Private, loopback, and link-local URLs are not allowed.")
        if isinstance(ip, ipaddress.IPv6Address):
            translated = ip.ipv4_mapped
            if ip in _NAT64_WELL_KNOWN:
                translated = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
            if translated is not None and not translated.is_global:
                raise ValueError("Translated private addresses are not allowed.")
        public_addresses.append(str(ip))
    if not public_addresses:
        raise ValueError("URL hostname did not resolve to a public address.")
    return parsed.hostname, public_addresses[0], port, parsed.scheme


def _validate_public_url(url: str) -> None:
    _resolve_public_url(url)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connect to a validated IP while retaining hostname TLS verification."""

    def __init__(self, hostname: str, ip: str, port: int, timeout: float) -> None:
        super().__init__(
            hostname,
            port=port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
        self._validated_ip = ip

    def connect(self) -> None:
        sock = socket.create_connection(
            (self._validated_ip, self.port),
            self.timeout,
        )
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def _request(url: str) -> tuple[int, dict[str, str], bytes]:
    hostname, ip, port, scheme = _resolve_public_url(url)
    parsed = urlparse(url)
    connection: http.client.HTTPConnection
    if scheme == "https":
        connection = _PinnedHTTPSConnection(hostname, ip, port, timeout=15)
    else:
        connection = http.client.HTTPConnection(ip, port=port, timeout=15)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    host_header = hostname if port in {80, 443} else f"{hostname}:{port}"
    try:
        connection.request(
            "GET",
            path,
            headers={
                "Host": host_header,
                "User-Agent": "basic-rag-reader/1.0",
                "Accept": "text/html,text/plain,application/xhtml+xml",
            },
        )
        response = connection.getresponse()
        body = response.read(_MAX_BYTES + 1)
        headers = {key.lower(): value for key, value in response.getheaders()}
        return response.status, headers, body
    finally:
        connection.close()


class WebPageReadInput(BaseModel):
    url: str = Field(min_length=8, max_length=2048)
    max_chars: int = Field(default=12_000, ge=500, le=20_000)


class WebPageReadTool(BaseAgentTool):
    name: str = "read_web_page"
    description: str = "Read bounded text from an explicit public HTTP(S) web page."
    args_schema: Type[BaseModel] = WebPageReadInput

    def _run(self, url: str, max_chars: int = 12_000) -> str:
        current_url = url
        try:
            for _ in range(_MAX_REDIRECTS + 1):
                status, headers, body = _request(current_url)
                if status in {301, 302, 303, 307, 308}:
                    location = headers.get("location")
                    if not location:
                        raise ValueError("Redirect did not include a destination.")
                    current_url = urljoin(current_url, location)
                    continue
                if status < 200 or status >= 300:
                    raise ValueError(f"Web page returned HTTP {status}.")
                content_type = headers.get("content-type", "").lower()
                if not any(kind in content_type for kind in ("text/", "html", "xml")):
                    raise ValueError("URL did not return readable text.")
                if len(body) > _MAX_BYTES:
                    raise ValueError("Web page exceeded the response-size limit.")
                raw = body.decode("utf-8", errors="replace")
                text = _extract_text(raw) or raw
                return self._format_success(
                    text[:max_chars],
                    metadata={"url": current_url, "truncated": len(text) > max_chars},
                ).to_str()
            raise ValueError("Too many redirects.")
        except (ValueError, OSError, ssl.SSLError, http.client.HTTPException) as exc:
            return self._format_error(str(exc)).to_str()

    async def _arun(self, url: str, max_chars: int = 12_000) -> str:
        return await asyncio.to_thread(self._run, url, max_chars)
