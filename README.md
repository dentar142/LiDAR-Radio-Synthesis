# LiDAR-Radio-Synthesis

LiDAR + blueprints → geometry / semantics → ray tracing → multi-physics correction → expert routing → radio synthesis.

```bash
pip install -e ".[runtime,radio,test]"
python tools/run_stage.py --config config/hkustgz_auto_reconstruction.example.yaml --stage all
python tools/run_radio.py predict --input data/band --output runs/band
```

| Interface | Input → Output |
| --- | --- |
| `tools/run_stage.py` | Scene YAML, textured mesh, blueprints → geometry, semantic/material mapping |
| `src.radio.legacy.aligned_factorial` | Sionna XML, receivers, alignment → RT features, calibrated experiments |
| `tools/run_radio.py` | `points.csv`, `rt.npz`, `configs.json` → predictions, routing audit |

[Input schema and RT commands](docs/radio-workflow.md) · [Scene config](config/hkustgz_auto_reconstruction.example.yaml) · [Data](docs/data-sources.md)

```bash
python tools/run_radio.py demo --output runs/demo
python -m pytest -q
```
