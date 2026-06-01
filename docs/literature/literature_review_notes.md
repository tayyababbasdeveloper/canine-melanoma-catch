# Literature Review Notes (Weeks 1–2)

Working notes refining the proposal's background section. For each paper:
**method used** + **key finding** + **why it justifies our approach**.

---

## A. Clinical & comparative-oncology background

| Reference | Method / scope | Key finding | Relevance to project |
|-----------|----------------|-------------|----------------------|
| Schadendorf et al. (2018), *The Lancet* | Review of melanoma | Melanoma = ~5% of skin cancers but ~75% of skin-cancer deaths | Motivates importance of accurate melanoma detection |
| Curtin et al. (2005), *NEJM* | Genetic analysis | Distinct mutation sets (KIT/NRAS vs BRAF) in UV vs non-UV melanoma | Justifies studying non-UV pathways |
| Gillard et al. (2014), *Pigment Cell & Melanoma Res.* | Canine melanoma genetics | Canine melanomas model **non-UV** human pathways | Core justification for canine model |
| Prouteau & André (2019), *Genes* | Clinical/histological/genetic comparison | Strong canine–human melanoma parallels | Supports comparative-oncology framing |
| Nishiya et al. (2016), *Vet. Sciences* | Review | Oral melanoma most common in dogs, no UV involvement | Translational relevance |

---

## B. The dataset

| Reference | Notes |
|-----------|-------|
| Fragoso-Garcia et al. (2023), **CATCH**, TCIA | 750 WSIs, expert-annotated, multiple cutaneous tumour subtypes. Our data source. |
| Clark et al. (2013), *J. Digital Imaging* | TCIA infrastructure — how the data is hosted/accessed. |

---

## C. Deep learning — classification

| Reference | Method | Finding | Why we use it |
|-----------|--------|---------|---------------|
| Esteva et al. (2017), *Nature* | Inception-v3, 129k images | Dermatologist-level skin-cancer classification, AUC 0.96 | Proof DL matches experts |
| He et al. (2016), CVPR | **ResNet** (residual connections) | Trains 150+ layer nets, mitigates vanishing gradient | Our classifier #1 (ResNet-50) |
| Tan & Le (2019), ICML | **EfficientNet** (compound scaling) | SOTA accuracy with fewer params | Our classifier #2 (EfficientNet-B3) |
| Hekler et al. (2019), *Eur. J. Cancer* | ResNet on 695 histo images | 93.1% accuracy, pathologist-level | Benchmark for melanoma F1 (~90%) |
| Brinker et al. (2019), *Eur. J. Cancer* | Inception-v4, 12k dermoscopy | Beat 136/157 dermatologists, 95% sens. | Evidence for transfer learning |
| Aubreville et al. (2020), *Sci. Data* | Canine histo, patch CNN + aggregation | >88% accuracy on canine tumours | Closest veterinary precedent |

---

## D. Deep learning — segmentation

| Reference | Method | Finding | Why we use it |
|-----------|--------|---------|---------------|
| Ronneberger et al. (2015), MICCAI | **U-Net** encoder-decoder + skip connections | Dice 0.92 with only 30 images | Our segmentation backbone |
| Oktay et al. (2018), arXiv | **Attention U-Net** (soft attention gates) | +3.6% Dice on pancreas | Variant we compare against |

---

## E. Preprocessing & interpretability

| Reference | Method | Why we use it |
|-----------|--------|---------------|
| Macenko et al. (2009), IEEE ISBI | Stain normalisation in OD space | Removes colour variation before training (implemented this fortnight) |
| Selvaraju et al. (2017), ICCV | **Grad-CAM** | Will visualise which regions drive classification (interpretability) |
| Simonyan & Zisserman (2015), ICLR | VGG | Context on transfer-learning backbones |

---

## Identified gap (restated)

Strong canine–human melanoma parallels + a curated dataset (CATCH) exist, **but**
few studies apply modern segmentation/classification DL specifically to **canine
melanoma**. This project fills that gap and tests translational relevance to human
non-UV melanoma.

---

## Citation style
Harvard (author–date), consistent with the proposal. Manage with a `.bib` file /
Zotero for the dissertation.
