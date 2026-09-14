# Release checklist

This repository publishes source code, example configuration, documentation and tests. Private measurements, scene meshes, photographs, credentials, generated runs and media are external assets and are never release inputs.

## Local verification

1. Install the geometry baseline with `python -m pip install -e ".[runtime,test]"`.
2. Run `python -m pytest -q`.
3. Run `python -m compileall -q src tests tools`.
4. Build the Python distributions with `python -m build` and verify `import src` from the wheel.
5. Run `python tools/build_release.py --output-dir release`.
6. Review `release/pi-razer-source.sha256.json` before distribution.

The release builder uses an allowlist and rejects path traversal, symlinks, binary/model/data formats, absolute user paths, private keys and common credential assignments. The resulting archive contains only repository source and supporting text files.

## Optional execution profiles

- `runtime` installs the plan/LiDAR reconstruction dependencies.
- `radio-core` installs the CPU measurement, expert-fitting and plotting dependencies used in CI.
- `radio-neural` installs PyTorch for the optional neural experts.
- `radio` installs both CPU and neural radio dependencies for complete local experiments.
- `rt` pins the report-aligned Sionna RT and Mitsuba versions and normally requires a compatible GPU environment.

The checked-in examples describe interfaces and expected inputs. The full scene run requires external measurement records and model assets, so public CI verifies deterministic unit contracts and packaging rather than executing `config/example_scene.yaml`.

## Before publication

- Confirm all empirical figures and reported metrics come from frozen experiment outputs.
- Record the commit identifier, environment lock information and input-data provenance.
- Keep transmitter configuration, measurement records and large geometry outside the archive.
- Add the public dataset URL only after its ownership, license and content have been verified.
- Do not publish from an unreviewed working tree.
