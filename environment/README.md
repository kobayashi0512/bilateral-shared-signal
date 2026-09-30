# Environment and compute notes

The analyses used Python-based data and vision tooling. A complete environment lockfile, container image, GPU driver record, and hardware inventory were not preserved, so this release does not claim bitwise or one-command reproducibility.

Verified analysis/runtime versions recorded for the project included Python 3.12.14, NumPy 2.2.1, pandas 2.2.3, SciPy 1.18.1, scikit-learn 1.6.0, PyTorch 2.5.1, torchvision 0.20.1, Pillow 11.0.0, and Matplotlib 3.10.0. Figure/document packaging also used a separate desktop runtime; these versions are not necessarily the environment in which every model experiment ran.

Use a clean environment and install compatible versions of the packages imported by the specific script. Confirm the torchvision ImageNet ResNet-50 V2 weight enum and preprocessing transform before feature extraction. Do not silently substitute a different model weight or transform for a frozen evaluation.

The source image data are not included. Compute time depends on storage and CPU/GPU; the feature extraction scripts are the expensive step, while manifest audits and aggregate figure generation are comparatively light. Local feature caches and checkpoints should remain outside the public repository.
