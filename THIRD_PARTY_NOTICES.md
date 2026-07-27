# Third-party notices

This source package does not redistribute third-party datasets, SAM2 source code, or pretrained weights.

External runtime components include:

- Segment Anything 2 and SAM2.1 checkpoints, obtained separately from the official Meta repository and used under their published license;
- PyTorch and torchvision;
- NumPy;
- Pillow;
- Hydra and OmegaConf;
- iopath;
- PyYAML;
- tqdm.

Users must obtain NUAA-SIRST, NUDT-SIRST, and IRSTD-1K from their respective maintainers and comply with the terms attached to each dataset.

The dependency declarations in `pyproject.toml` are provided for reproducibility and do not alter the licenses of those projects.
