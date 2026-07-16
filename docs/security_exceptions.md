# Security Exceptions

## PYSEC-2026-3447: setuptools

- Status: Temporarily accepted
- Package: `setuptools==81.0.0`
- Severity: Medium
- Recorded: 2026-07-16
- Review by: 2026-08-16

### Reason

PyTorch 2.12.1 requires `setuptools<82`, while the advisory is fixed in
setuptools 83.0.0. Upgrading now would violate the validated PyTorch dependency
contract and cause `pip check` to fail.

### Applicability

The advisory concerns Unicode filename normalization while building source
distributions on macOS APFS or HFS+. This project is currently validated on
Linux x86_64, uses editable or wheel-based installation, and does not publish
source distributions.

### Mitigations

- Do not build or publish source distributions with this environment.
- Accept project files only from trusted sources.
- Continue auditing with an explicit exception for `PYSEC-2026-3447`.
- Keep `pip check`, the complete test suite, and CI dependency installation
  passing.

### Removal Conditions

Remove this exception and upgrade setuptools when PyTorch supports
`setuptools>=83`, before adding macOS source-distribution publishing, or when
the review date is reached.
