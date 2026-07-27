# Anonymization report

## Release boundary

Included:

- SPARK-SAM model, training losses, and four-phase training chain;
- prompt-estimator training and validation selection;
- SAM2.1-Large response-guidance cache construction;
- calibration and dense prompt cache construction;
- NUAA-SIRST, NUDT-SIRST, and IRSTD-1K dataset adapters and split identifiers;
- validation operating-point selection, locked test evaluation, runtime benchmarking, metrics, and tests.

Excluded:

- datasets and annotations;
- official SAM2 source and checkpoints;
- model checkpoints, caches, masks, logs, and result tables;
- obsolete experiment branches, unrelated analysis scripts, and deployment experiments;
- credentials, shell history, private configuration files, repository metadata, and infrastructure documentation.

## Semantic-name conversion

Internal experiment labels and numbered stages were replaced by method-level names:

- prompt estimator;
- response guidance;
- joint adaptation;
- response calibration;
- false-alarm calibration;
- high-resolution refinement;
- calibration response cache.

Version-number model aliases in the prompt estimator were replaced by architecture descriptions such as `feature_pyramid` and `high_resolution_feature_pyramid`.

## Path handling

No machine path is embedded in the release. Runtime locations are provided by:

- `SAM2_REPO`
- `SAM2_CHECKPOINT_ROOT`
- `ARTIFACT_ROOT`
- `NUAA_SIRST_ROOT`
- `NUDT_SIRST_ROOT`
- `IRSTD_1K_ROOT`

Training artifacts use project-relative paths under `artifacts/`. The split manifest contains dataset identifiers and sample identifiers only.

## Audit checks

Before packaging, the release is scanned for:

- author and machine identifiers;
- server aliases, private hostnames, and private-network addresses;
- Windows user paths and Unix home/project paths;
- removed internal experiment labels and numbered stage aliases;
- Python cache files, checkpoints, archives, logs, and generated artifacts.

The final archive contains a `MANIFEST.txt` with per-file SHA-256 hashes. The archive checksum is distributed beside the archive.
