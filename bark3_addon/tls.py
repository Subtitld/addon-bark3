"""Which CA certificates the downloads trust, in the frozen build.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The frozen bundle carries the OpenSSL of the Python that froze it, and that
OpenSSL looks for CA certificates where the build machine keeps them: on
Linux, /usr/lib/ssl for a Python built on Ubuntu, which Fedora and openSUSE do
not have; on macOS, a folder inside the build's Python.framework, which no
user's Mac has. The download then fails with "CERTIFICATE_VERIFY_FAILED:
unable to get local issuer certificate".

So the frozen build points SSL_CERT_FILE at the CA bundle this machine's
system keeps: the system store, with any certificate an administrator added.
An SSL_CERT_FILE or SSL_CERT_DIR already set always wins (a proxy that
inspects HTTPS needs its own). Windows is left alone: there Python reads the
system's certificate store.
"""

from __future__ import annotations

import os
import sys

# Where systems keep their CA bundle: the list Go's crypto/x509 checks.
SYSTEM_CA_FILES = (
    '/etc/ssl/certs/ca-certificates.crt',                 # Debian, Ubuntu, Arch, Gentoo
    '/etc/pki/tls/certs/ca-bundle.crt',                   # Fedora, RHEL
    '/etc/ssl/ca-bundle.pem',                             # openSUSE
    '/etc/pki/tls/cacert.pem',                            # OpenELEC
    '/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem',  # CentOS, RHEL 7
    '/etc/ssl/cert.pem',                                  # macOS, Alpine
)


def configure_ca_bundle(frozen: bool | None = None, platform: str | None = None) -> str | None:
    """Set SSL_CERT_FILE for the frozen Linux and macOS builds. Returns the bundle chosen, or None."""
    frozen = getattr(sys, 'frozen', False) if frozen is None else frozen
    platform = sys.platform if platform is None else platform
    if not frozen or not (platform.startswith('linux') or platform == 'darwin'):
        return None
    if os.environ.get('SSL_CERT_FILE') or os.environ.get('SSL_CERT_DIR'):
        return None
    for path in SYSTEM_CA_FILES:
        if os.path.isfile(path):
            os.environ['SSL_CERT_FILE'] = path
            return path
    return None
