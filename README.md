# Exploring galaxy mergers using computer vision

Code for my BSc thesis in astronomy (University of Groningen, Kapteyn Astronomical Institute, 2026; supervisor Dr. Ting-Yun Cheng).

The thesis asks how much merger information can be recovered from galaxy images without merger labels. Two self-supervised contrastive encoders, SimCLR and NNCLR, are trained on the IllustrisTNG mock images of the Galaxy Merger Challenge (Margalef-Bentabol et al. 2024, "MB24") and evaluated in two ways: a linear probe on the frozen features, and a fully label-free Gaussian-mixture clustering. The trained encoders are then applied without retraining to Horizon-AGN mocks and real HSC observations.

| | SimCLR | NNCLR |
|---|---|---|
| TNG linear-probe AUC | 0.782 | 0.751 |
| TNG clustering AUC (z-conditioned, K = 64) | 0.740 | 0.738 |
| Horizon-AGN zero-shot AUC | 0.608 | 0.573 |
| HSC visual sample zero-shot AUC (clear mergers vs clear non-mergers) | 0.832 | 0.806 |

For reference, the six supervised methods of MB24 reach 0.765–0.851 on TNG.

## Layout

```
data/
  prepare_transfer_data.py   Horizon-AGN / HSC FITS -> 128 px stacks + label arrays
  README.md                  how the TNG stack was built
training/
  common.py                  encoder, augmentations and data loading shared by both arms
  config_simclr.py, simclr_arm.py, run_simclr.sh   SimCLR: train, extract features, PCA
  config_nnclr.py,  nnclr_arm.py,  run_nnclr.sh    NNCLR:  same
  encode_stack.py            apply a trained encoder to any image stack (Horizon-AGN, HSC)
notebooks/
  results.ipynb              linear probes and transfer: Tables 4.1-4.7, Figure 4.1
  clustering.ipynb           GMM sweep and cluster anatomy: Table 4.8, Figures 4.2-4.6
```

## Notes

The data are not included; they were obtained directly from the MB24 team.

## References

- Margalef-Bentabol, B., et al. 2024, A&A, "Galaxy merger challenge: A comparison study between machine learning-based detection methods"
- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. 2020, "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR)
- Dwibedi, D., Aytar, Y., Tompson, J., Sermanet, P., & Zisserman, A. 2021, "With a Little Help from My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations" (NNCLR)
