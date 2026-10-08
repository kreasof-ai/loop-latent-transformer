# Verification

Run `python -m unittest discover -s tests -v` from the repository root with Torch
and Tensor's Torch adapter installed. These checks execute on CPU.

The suite checks relocated payload SHA256 values, resolution of recorded paths,
numerical source preservation, checkpoint gradients, cached/full-forward logits,
serving-copy refresh, and exact stack/fixed-depth parameter matching. It verifies
behavior affected by code movement without rerunning the GPU performance study.

Strict GPU artifact and case-grid auditing lives in the experiment summary tools.
See [reproducibility](../docs/REPRODUCIBILITY.md).

Added-family CPU tests compare none/AC gradients for every parameter, cached/full-
forward supplied-token logits, rewind behavior, architecture-specific work counts, and seeded stochastic GRT
checkpoint gradients/RNG consumption.
