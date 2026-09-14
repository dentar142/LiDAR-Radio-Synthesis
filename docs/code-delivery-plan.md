# Radio-twin code integration

## Scope

Preserve the existing plan/LiDAR/material reconstruction implementation and integrate the downstream radio workflow used by the aligned report. No private measurements, meshes, scene credentials, or remote deployment are included. Remote publication is separate from local packaging.

## Acceptance criteria

1. Existing geometry contract and reconstruction tests still pass.
2. Exact experiment dependencies are included or explicitly identified, with origin hashes.
3. Public commands use configuration/arguments rather than machine-specific paths.
4. CPU smoke tests cover fit-label isolation, spatial splits, all ten experts, selection, scoring and repeatability where supported.
5. RT uses the existing Sionna implementation, with private input assets external and a separate GPU dependency profile. Synthetic fixtures are labelled, never reported as empirical accuracy.
6. Documentation covers scene preparation, measurement adaptation, RT features, calibration, fitting, selection, evaluation, resume and outputs.
7. Packaging includes source/config/docs/tests only and scans for absolute local paths, credentials and private data. No push is performed.

## Sequence

- Capture the clean repository baseline and execute existing tests before changing behavior.
- Locate and audit the frozen H18 dependency chain and aligned calibration implementation.
- Integrate in an isolated radio module, keeping existing geometry interfaces stable.
- Add portability wrappers, explicit data contracts and configuration examples.
- Run regression tests, CLI smoke tests, source compilation and a safe release inventory.
- Produce a local release archive and verification report with any GPU/private-data validation gaps.
