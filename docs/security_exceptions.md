# Security Exceptions

## Current Status

No active security exceptions.

The dependency audit runs without `--ignore-vuln` entries. Temporary
exceptions are removed as soon as compatible corrected releases pass the
project's training, evaluation, inference, and CI validation.

## Resolved on 2026-09-09

- Upgraded PyTorch from 2.12.1 to 2.13.0, removing the
  `PYSEC-2025-194` exception.
- Upgraded setuptools from 81.0.0 to 84.0.0, removing the
  `PYSEC-2026-3447` exception.
- Upgraded GitPython from 3.1.53 to 3.1.62, removing the four temporary
  GitPython advisory exceptions.
- Upgraded MLflow to 3.16.0 and raised safe lower bounds for aiohttp,
  cryptography, httpcore2, httpx2, and sqlparse.

The resolved dependency set passes `pip check` and `pip-audit` without
ignored vulnerability identifiers.

## Audit Scope

The local GPU environment installs `torch==2.13.0+cu130` from PyTorch's CUDA
package index. Because this local-version wheel is not published on PyPI,
`pip-audit` reports it as an unaudited external distribution. The project
and CI dependency contracts require the corrected 2.13 release line, while
the remaining PyPI packages are audited normally.
