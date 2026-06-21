# Examples

This repository does not include real subject-level fMRI data. The Quick Start
uses synthetic data generated at runtime by `quickstart/quickstart_smoke.py` so
new users can verify the full pipeline without dataset access or privacy
concerns.

The smoke test creates:

```text
outputs/quickstart_smoke/
  raw/sample_fmri.nii.gz
  processed_npz/sample_fmri_seg000.npz
  features/sample_fmri_seg000_tokens.npz
```

Use the generated files only as a format and pipeline check. They are not a
scientific dataset and should not be used for model evaluation.
