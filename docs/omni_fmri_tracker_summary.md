# Omni-fMRI Tracker-Compatible Summary

This table mirrors the NeuroSTORM tracker schema:

```text
Big parts | detailed part | Neurostorm old | Neurostorm updated | Omni-fMRI
```

Numerical Omni values are pending until real UKB Omni extraction, GWAS, LDSC,
ENIGMA, and structural mapping jobs finish. Use
`scripts/omni_pipeline/summarize_omni_results.py` to regenerate a TSV version
from completed output files.

| Big parts | detailed part | Neurostorm old | Neurostorm updated | Omni-fMRI |
| --- | --- | --- | --- | --- |
| Embedding And GWAS Inputs | Embedding extraction | Completed in NeuroSTORM tracker | Completed in NeuroSTORM tracker | Planned. Use `prepare_ukb_manifest.py` and `extract_omni_embeddings.py`; output `eid + emb_001..emb_768`. |
| Embedding And GWAS Inputs | GWAS input preparation | Completed for 288 embeddings | Completed for 288 embeddings | Planned. Use `prepare_gwas_inputs.py`; enforce `FID=IID=pure_UKB_eid`; RankINT all 768 embeddings. |
| Embedding And GWAS Inputs | PLINK2 GWAS screening route | Completed screening route | Completed screening route | Planned. Use `submit_plink2_gwas.pbs` as 1-768 PBS array. Screening only, not final discovery. |
| Embedding And GWAS Inputs | SAIGE / mixed-model GWAS | Pending / collaborator route | Pending / collaborator route | Planned handoff. Reuse RankINT phenotypes and covariates; exact SAIGE schema TODO. |
| SNP Heritability / LDSC | LDSC h2 completion | Completed for 288 | Completed for 288 | Planned for 768. Use `submit_ldsc_h2.pbs` after merged PLINK2 sumstats exist. |
| SNP Heritability / LDSC | Mean h2 / range / intercept | See NeuroSTORM tracker | See NeuroSTORM tracker | Pending numeric results. |
| SNP Heritability / LDSC | Top h2 embeddings | See NeuroSTORM tracker | See NeuroSTORM tracker | Pending numeric results; compare distributions, not embedding indices. |
| Loci And Discovery Signal | P < 5e-8 unique regions | See NeuroSTORM tracker | See NeuroSTORM tracker | Planned. Use `count_loci.py`; threshold `P < 5e-8`. |
| Loci And Discovery Signal | Strict unique regions | See NeuroSTORM tracker | See NeuroSTORM tracker | Planned. Strict threshold `P < 5e-8 / 768`. |
| Loci And Discovery Signal | Top locus-count embeddings | See NeuroSTORM tracker | See NeuroSTORM tracker | Pending numeric results. |
| Benchmark and comparison | Omni vs NeuroSTORM old / updated | Model-level reference | Model-level active comparison | Planned. Compare h2 distributions, loci counts, strict loci, genomic regions, Big40/Zhao novelty. Do not compare `emb_001` across models. |
| Structural Brain-IDP Mapping | All embedding structural mapping | Selected anchors / old mapping | All-288 updated mapping available | Planned all-768 Omni mapping. Reuse NeuroSTORM structural mapping design; add ICV/head-size sensitivity before strong morphology claims. |
| ENIGMA Genetic Correlation Validation | ENIGMA rg grid | Completed 288 x 77 | Completed 288 x 77 | Planned 768 x 77 = 59,136 tests, using existing ENIGMA 77 usable traits if available. |
| Integrated priority table | Priority embedding table | Partial / old anchors | Pending integrated table | Planned. Use `build_priority_table.py` with h2, loci, ENIGMA, structural, novelty, and cluster inputs. |
