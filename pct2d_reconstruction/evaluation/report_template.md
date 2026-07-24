# Experiment {{ experiment_id }} common evaluation

## Acquisition and executed configuration

- Simulation configuration: `{{ simulation_config }}`
- Preprocessing configuration: `{{ preprocessing_config }}`
- Reconstruction configuration: `{{ executed_reconstruction_config }}`
- Frozen artifact manifest: `{{ baseline_manifest }}`

Always distinguish the planned configuration from the configuration recorded by
the completed run. Report the latter as authoritative.

## Data split

Report the versioned identity/hash rule, seed, total/train/validation counts,
per-angle coverage, and split-manifest hash. Training data may update an
algorithm; validation data may only evaluate fixed checkpoints.

## Common RSP results

Use the columns from `checkpoint_metrics.csv` without redefining ROIs:

| Checkpoint | Water mean | Water std | Phantom RSP RMSE | Material platform recovery | ROI CNR | Edge 10-90 | Validation WEPL RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| {{ checkpoint }} | {{ water_mean_rsp }} | {{ water_std_rsp }} | {{ phantom_rsp_rmse }} | {{ material_recovery }} | {{ roi_cnr }} | {{ edge_width_mm }} | {{ validation_wepl_rmse_mm }} |

State unavailable metrics explicitly. Do not substitute composition RED for
the declared RSP truth and do not compare online training residuals with fixed
validation-checkpoint residuals as though they were the same quantity.

## Resources and acceptance

Report elapsed time, CPU/GPU, peak memory, throughput, finite-value checks,
artifact/split hash verification, deviations from plan, and the decision on
whether the next research stage may start.

