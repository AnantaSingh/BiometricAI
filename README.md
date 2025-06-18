# 🔍 Attacking Face Recognition Systems Using Deepfakes

**Author:** Ananta Singh  
**Affiliation:** Department of Computer Science, San Jose State University  
**Contact:** ananta.singh@sjsu.edu

## 📘 Overview

This project investigates the vulnerability of facial recognition (FR) systems to deepfake attacks. It involves:
- Building an FR system using **FaceNet** and three classifiers (SVM, MLP, Random Forest).
- Generating deepfakes using **InsightFace** and **SimSwap**.
- Evaluating the misidentification rate of deepfakes on the FR system.

---

## 🗂️ Dataset

- **Source:** [PubFig Dataset](https://www.kaggle.com/datasets/kaustubhchaudhari/pubfig-dataset-256x256-jpg)
- **Classes:** 6 Celebrities – Lindsay Lohan, Tom Cruise, Leonardo DiCaprio, Daniel Radcliffe, Orlando Bloom, Miley Cyrus
- **Total Images:** 1,667 (after filtering)
- **Augmentation:** +8,334 images → Total: 10,001  
  Augmentations include affine transformations, color and brightness changes, noise, blur, dropout, and perspective shifts.

---

## 🧠 Face Recognition System

- **Preprocessing:**  
  - Face detection via MTCNN  
  - Image resizing to 160×160  
  - Normalization and duplicate filtering via hash checking

- **Feature Extraction:**  
  - **FaceNet** (InceptionResnetV1 pretrained on VGGFace2)  
  - PCA reduced 128D embeddings to 30D

- **Classifiers Used:**  
  - Support Vector Machine (SVM) – **87% accuracy**  
  - Multi-Layer Perceptron (MLP) – 85% accuracy  
  - Random Forest – 84% accuracy

---

## 🧪 Deepfake Generation

### InsightFace
- Used `buffalo_l` model + `inswapper_128.onnx` for face-swapping
- Tools: RetinaFace (for alignment), ArcFace embeddings
- **Pairs Swapped:** Leonardo DiCaprio ↔ Lindsay Lohan, Orlando Bloom ↔ Tom Cruise

### SimSwap
- Used original SimSwap repo with extended automation
- Identity preserved via ArcFace + ID Injection Module
- **Pairs Swapped:** Leonardo DiCaprio ↔ Orlando Bloom, Lindsay Lohan ↔ Tom Cruise

---

## 🧬 Evaluation Results

### Misidentification Rates:
| Deepfake Source | Correct Predictions | Misidentification Rate |
|------------------|----------------------|--------------------------|
| InsightFace      | 39/40                | **0.03**                 |
| SimSwap          | 20/39                | **0.49**                 |

- Visual analysis included: confusion matrices, ROC curves, t-SNE plots
- InsightFace performed better due to higher alignment fidelity

---

## ⚠️ Challenges Addressed

- Overfitting mitigated with class balancing, augmentation, and PCA
- Memory issues handled with batching and garbage collection
- Face detection failures logged and managed
- Duplicate and leakage prevention using image hashing

---

## 📌 Future Work

- Integrate deepfake detection modules
- Extend dataset to include more subjects and settings
- Evaluate newer deepfake models (e.g., `inswapper_dax`, `inswapper_cyn`)

---

## 🧾 References

- [FaceNet](https://arxiv.org/abs/1503.03832)  
- [SimSwap Paper](https://arxiv.org/abs/2106.06340)  
- [InsightFace GitHub](https://github.com/deepinsight/insightface)  
- [PubFig Dataset](https://www.kaggle.com/datasets/kaustubhchaudhari/pubfig-dataset-256x256-jpg)

---

## 🛠️ How to Run

1. Clone the repository and install dependencies from `requirements.txt`
2. Prepare dataset as outlined in `data_preprocessing.py`
3. Run augmentation with `augment_images.py`
4. Train classifiers using `train_classifiers.py`
5. Generate deepfakes with `generate_deepfakes.py`
6. Evaluate robustness via `test_deepfakes.py`

---

> ⚠️ *This project is intended for academic research and ethical use only. Misuse of deepfake technologies can lead to serious ethical and legal consequences.*
