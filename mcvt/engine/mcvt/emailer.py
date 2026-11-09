"""Email delivery of journey videos over SMTP (FR-ED-01 .. FR-ED-04).

Credentials are read from the environment (SMTP_USER / SMTP_PASSWORD, usually
via the .env file) - never from source code (NFR-S-01).
"""

from __future__ import annotations

import logging
import mimetypes
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

log = logging.getLogger("email")


class EmailService:
    def __init__(self, host: str, port: int, username: str | None = None, password: str | None = None,
                 security: str = "starttls", sender: str | None = None, timeout: float = 60,
                 max_attachment_mb: float = 25):
        self.host, self.port = host, int(port)
        self.username = username if username is not None else os.environ.get("SMTP_USER", "")
        self.password = password if password is not None else os.environ.get("SMTP_PASSWORD", "")
        self.security = (security or "starttls").lower()
        self.sender = sender or self.username
        self.timeout = timeout
        self.max_bytes = int(float(max_attachment_mb) * 1024 * 1024)
        self.attachment: Path | None = None
        self.last_error: str | None = None
        self.recipient: str | None = None

    @classmethod
    def from_config(cls, cfg) -> "EmailService":
        return cls(
            host=os.environ.get("SMTP_HOST") or cfg.get("email.host"),
            port=int(os.environ.get("SMTP_PORT") or cfg.get("email.port", 587)),
            security=cfg.get("email.security", "starttls"),
            sender=cfg.get("email.sender") or None,
            timeout=cfg.get("email.timeout", 60),
            max_attachment_mb=cfg.get("email.max_attachment_mb", 25),
        )

    def attachVideo(self, video_file: str | Path) -> bool:
        """Remember the video to attach (FR-ED-01)."""
        p = Path(video_file)
        if not p.is_file():
            self.last_error = f"Video file {p} does not exist"
            log.error(self.last_error)
            self.attachment = None
            return False
        self.attachment = p
        return True

    attach_video = attachVideo

    def sendEmail(self, recipient: str | None = None, subject: str = "MCVT journey video", body: str = "") -> bool:
        """Send the message; returns True on success, False on any failure (FR-ED-02/03/04)."""
        self.last_error = None
        recipient = recipient or self.recipient
        try:
            if not recipient:
                raise ValueError("No recipient address given")
            if not self.username or not self.password:
                raise smtplib.SMTPAuthenticationError(535, b"SMTP_USER / SMTP_PASSWORD are not configured")
            msg = EmailMessage()
            msg["From"] = self.sender or self.username
            msg["To"] = recipient
            msg["Subject"] = subject
            msg.set_content(body or "Please find the requested journey video attached.")
            if self.attachment is not None:
                size = self.attachment.stat().st_size
                if size > self.max_bytes:
                    raise ValueError(f"Video is {size / 1048576:.1f} MB, larger than the "
                                     f"{self.max_bytes / 1048576:.0f} MB attachment limit")
                ctype = mimetypes.guess_type(self.attachment.name)[0] or "video/mp4"
                maintype, subtype = ctype.split("/", 1)
                msg.add_attachment(self.attachment.read_bytes(), maintype=maintype, subtype=subtype,
                                   filename=self.attachment.name)
            context = ssl.create_default_context()
            if self.security == "ssl":
                server = smtplib.SMTP_SSL(self.host, self.port, timeout=self.timeout, context=context)
            else:
                server = smtplib.SMTP(self.host, self.port, timeout=self.timeout)
            with server:
                if self.security == "starttls":
                    server.starttls(context=context)
                server.login(self.username, self.password)
                server.send_message(msg)
            log.info("Email with %s sent to %s", self.attachment.name if self.attachment else "no attachment", recipient)
            return True
        except smtplib.SMTPAuthenticationError as exc:
            self.last_error = "SMTP login failed - check SMTP_USER / SMTP_PASSWORD (" + _text(exc) + ")"
        except (smtplib.SMTPException, OSError, ValueError) as exc:
            self.last_error = f"Email could not be sent: {_text(exc)}"
        log.error(self.last_error)
        return False

    send_email = sendEmail


def _text(exc: Exception) -> str:
    if isinstance(exc, smtplib.SMTPResponseException):
        err = exc.smtp_error.decode(errors="replace") if isinstance(exc.smtp_error, bytes) else str(exc.smtp_error)
        return f"{exc.smtp_code} {err}"
    return str(exc)
