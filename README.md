# Multimodal Sexism Detection in Memes — EXIST 2026

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)

This is our submission to the memes track of **[EXIST 2026](https://nlp.uned.es/exist2026/)** (sEXism Identification in Social neTworks), a shared task at [CLEF 2026](https://clef2026.clef-initiative.eu/labs/exist/). The 2026 edition adds a new kind of input: besides each meme's text and image, it provides the **physiological responses of the people who viewed it**, recorded as EEG, eye tracking and heart rate.

We build multimodal deep-learning models that combine all three sources in English and Spanish to solve three subtasks:

| Subtask | Question | Type |
|---|---|---|
| 2.1 Sexism identification | Is the meme sexist? | Binary (YES / NO) |
| 2.2 Source intention | Is the sexism direct, or does it judge/criticise sexism? | Binary (DIRECT / JUDGEMENTAL) |
| 2.3 Sexism categorization | Which types of sexism appear? | Multi-label (5 categories) |

---

## Approach

Each meme passes through three branches that are then fused:

- **Text:** XLM-RoBERTa for English and [RoBERTuito](https://huggingface.co/pysentimiento/robertuito-base-uncased) for Spanish. RoBERTuito is pre-trained on Spanish social-media text.
- **Image:** Vision Transformer (ViT-Base, 16×16 patches).
- **Physiological signals:** sensor features averaged across viewers, encoded by an MLP and projected to the same 768-dimensional space as text and image.

We compared two fusion strategies:

| | **Cross-attention fusion** | **Gated fusion** |
|---|---|---|
| Idea | All text tokens, image patches and sensor tokens attend to each other, followed by a 2-layer Transformer | A learned gate gives each modality a weight between 0 and 1 and combines them as a weighted sum |
| Encoders | Frozen | Fine-tuned |
| Output | Softmax (2.1, 2.2) / Sigmoid (2.3) | Sigmoid |

**Labels:** the 6 annotators' votes are aggregated by majority vote (union of categories for 2.3). Memes without a majority are discarded. We used an 80/20 train/dev split per language.

## Results (F1 on our dev split)

| Fusion | Text encoders | 2.1 EN | 2.1 ES | 2.2 EN | 2.2 ES | 2.3 EN | 2.3 ES |
|---|---|---|---|---|---|---|---|
| Cross-attention | XLM-R only | 0.66 | 0.74 | 0.72 | 0.79 | 0.35 | 0.39 |
| Cross-attention | XLM-R + RoBERTuito | 0.67 | 0.70 | **0.80** | **0.88** | 0.38 | 0.49 |
| Gated | XLM-R only | 0.65 | 0.70 | 0.71 | 0.68 | 0.36 | 0.46 |
| Gated | XLM-R + RoBERTuito | **0.68** | **0.72** | **0.80** | 0.71 | 0.37 | **0.55** |

- **A language-specific encoder helps.** Using RoBERTuito for Spanish improves most scores, for example 0.79 → 0.88 in Spanish source intention.
- **Each fusion strategy wins on different subtasks.** Gated fusion is best for identification and multi-label categorization. Cross-attention is best for source intention. The final submission uses **gated fusion for 2.1 and 2.3** and **cross-attention for 2.2**.
- **Verbalising images did not work.** We first tried an approach that describes each image with Qwen-VL, merges the description with the meme text and classifies the result. It reached only 0.46 / 0.63 F1 on 2.1, so we discarded it. Turning an image into a text description seems to lose visual details that matter.

The predictions we submitted to the official test set are in [`results/official_run/`](results/official_run).

## Repository structure

```
├── exist2026_pipeline.ipynb   # end to end: data → train & compare → predict test → submission files
├── src/
│   ├── data.py                # label aggregation, splits, sensor feature extraction, PyTorch dataset
│   ├── models.py              # CrossAttentionFusionModel, GatedFusionModel
│   ├── train.py               # training loop, evaluation, prediction
│   └── submission.py          # EXIST-format JSON files
└── results/official_run/      # our submitted predictions
```

## How to run

1. Request the **EXIST 2026 dataset** from the [organisers](https://nlp.uned.es/exist2026/). It is not redistributed here. Unzip it so that this folder exists: `data/EXIST 2026 Memes Dataset/` (with `training/` and `test/` inside).
2. Install the dependencies and open the notebook. A GPU is recommended, and Google Colab works.

```bash
pip install -r requirements.txt
jupyter notebook exist2026_pipeline.ipynb
```

## Team

Course project for the Data Science degree at Universitat Politècnica de València (2025–26), developed by **Carlota Gui Escamilla** and Pablo Gil Martínez.
