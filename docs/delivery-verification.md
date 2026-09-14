# Local delivery verification — 2026-09-14

## Delivered

Geometry reconstruction remains in its original module. The H18 runtime dependency closure and aligned material/power-calibration controller are integrated under src/radio/legacy. Public adapters, strict point/RT alignment checks, an independent scorer, a synthetic executable example, optional dependency groups, CI and a source-only release builder are included.

The reviewed source was published to the repository's `main` branch on 2026-09-14. Local release archives are reproducible packaging artifacts and are not committed.

## Executed checks

- Existing geometry and contract tests passed before changes.
- The integrated test suite passed all 15 tests, covering reconstruction, contracts, source provenance, training-label isolation, supported-power calibration, expert selection, ID alignment and release safety.
- The complete synthetic CPU example ran twice using actual ten-expert predictions, three-fold MATCHED validation, selection, risk weights and independent scoring.
- Both runs produced byte-identical prediction CSV, out-of-fold CSV, audit JSON and software-test metric CSV.
- Python source compilation, command-line help and Git whitespace checks passed.
- Wheel packaging and isolated installed-package imports were tested during integration.
- Release archives are scanned for private paths/credentials and exclude measurements, model binaries and run caches. Their external SHA-256 manifest records every packaged file.

## Interpretation

The example uses analytic synthetic path gains and quick GP fitting. Its numerical errors are software-check outputs, not empirical results or substitutes for report measurements.

The complete private 240-task / 480-arm campaign and fresh GPU ray tracing were not rerun as part of local code packaging. Their execution requires the original normalized measurements, frozen preparation inputs, matching scene XML/meshes, alignment metadata and RT environment.

Some older compatibility scripts retain optional external legacy-runner arguments. The aligned fresh-source RT path uses the integrated Sionna implementation and does not invoke that legacy-runner option.

No source code is silently represented as the original binary-identical release: original hashes are provenance, while runtime and archive hashes describe the distributed files.
