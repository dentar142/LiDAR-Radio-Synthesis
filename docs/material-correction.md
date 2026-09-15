# Material correction candidates

Two experimental candidates complement the fixed-material workflow: `FIELD_RT_VOM` fits a continuous material field; `CLASS_RT_VOM_U2` fits class-shared materials and adds a local residual correction. VOM here is a line-integral attenuation proxy.

Install the `radio` and `rt` dependency groups. Provide a Sionna scene and material-prior JSON compatible with the source fitting routines. The input NPZ contains exactly:

| Key | Shape |
| --- | --- |
| `fit_xyz`, `fit_y`, `fit_id` | `(N,3)`, `(N,)`, `(N,)` |
| `query_xyz`, `query_id` | `(Q,3)`, `(Q,)` |
| `tx` | `(3,)` |

Coordinates are local metres; fit labels are dBm. Fit/query IDs and positions must be disjoint. Query labels are not accepted.

```bash
python -m src.radio.material_adaptive.material_expert_replacement --input fit-query.npz --output runs/material-a --scene scene.xml --priors priors.json --band n41 --expert FIELD_RT_VOM --steps 60 --samples 50000
```

Use a fresh output directory. Outputs are `prediction.npz` and `status.json`. For candidate B, select `CLASS_RT_VOM_U2`.

These candidates are separate from the released ten-expert predictor. Nested routing integration and GPU accuracy reevaluation are not included in this release's validation. Material fitting must be repeated within each training fold; outer-test labels must not influence fitting or expert replacement. Source provenance is recorded in `src/radio/material_adaptive/provenance.json`.
