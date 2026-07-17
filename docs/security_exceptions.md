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

## PYSEC-2025-194: torch

- Status: Temporarily accepted pending advisory reconciliation
- Package: `torch==2.12.1`
- Severity: Low to Medium
- Recorded: 2026-07-17
- Review by: 2026-08-17

### Reason

The default PyPI vulnerability service used by `pip-audit` reports
`torch==2.12.1` as affected and identifies 2.13.0 as the fixed version.
However, the GitHub-reviewed advisory reports affected versions through
2.12.0 and does not identify a patched version. The upstream vulnerability
metadata is therefore inconsistent.

Upgrading to PyTorch 2.13 requires a separate compatibility validation for
CUDA dependencies, model serialization, training, evaluation, inference, and
the Docker image.

### Applicability

The advisory concerns local memory corruption through `torch.jit.script`.
This project does not use TorchScript, `torch.jit.script`, or user-provided
model compilation. The current classifier uses an eager `nn.Module` and loads
artifacts produced by the controlled local training pipeline.

### Mitigations

- Do not introduce `torch.jit.script` or TorchScript while this exception is
  active.
- Load model and vectorizer artifacts only from the trusted training pipeline.
- Do not accept uploaded or externally supplied model artifacts.
- Keep training, inference, tests, dependency checks, and security audits
  passing.
- Track PyTorch 2.13 as a separate compatibility and dependency upgrade.

### Removal Conditions

Remove this exception after the upstream advisory metadata is reconciled and
PyTorch 2.13 is validated, or before introducing TorchScript, external model
artifacts, or user-controlled model loading.
