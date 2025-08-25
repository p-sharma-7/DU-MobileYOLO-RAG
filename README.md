# Explainable Marine Object Detection Using DU-MobileYOLO and Retrieval-Augmented Generation

### Overview
This project, MarineExplainAI, introduces a fully local, open-source pipeline that integrates real-time object detection with explainable vision-language reasoning for marine species monitoring.

We developed a hybrid architecture combining:
* DU-MobileYOLO → Lightweight, deformable CNN + MobileViT backbone for species detection.
* LLaVA-1.5 (7B) → Vision-language model for semantic explanations.
* Retrieval-Augmented Generation (RAG) → BM25 and quantum similarity reranking based local marine knowledge base to ground explanations in real facts.
* Achieved mAP@0.5 = 89.05% on challenging brackish underwater datasets.
* Generated 3,200+ domain-aware explanations of marine detections.
* Built a scalable, local pipeline deployable on research servers and edge devices.

<img width="1603" height="675" alt="image" src="https://github.com/user-attachments/assets/b0082eaa-7974-42aa-b1d4-e120dfc52a3a" />

### Datasets
Our pipeline leverages a combination of real-world and synthetic data sources:

**1. URPC2019 Dataset**

- Public underwater dataset with labeled images of marine species.
- Used for initial training of DU-MobileYOLO.

**2. Brackish Marine Dataset**
- Curated ~10,000 underwater frames from estuarine habitats.
- 3,211 high-confidence detections across 6 classes: fish, crab, starfish, jellyfish, sea cucumber, sea anemone.
- Split: 80% training, 20% validation.

**3. Synthetic Morphology Dataset (Vector DB for RAG)**
- Created due to limited morphology data.
- Species names extracted from OBIS dataset.
- Morphological traits generated using OpenBioLLM-70B (HuggingFace).
- ~79 JSON entries containing attributes like color, body shape, habitat, size, diet, notable features.
- Structured as a CSV with fields: species_name, class_id_label, class_name_label, response, prompt.


### Inferencing
example:
- **auto generated prompt :** What marine species is likely visible in this scene containing a crab?
- **explanation:** The image likely contains a crab because it displays characteristics consistent with crab habitats—such as shallow ocean waters and rocky surfaces. The visible exoskeleton, claws, and compact body shape align with known features of crabs. Additionally, the surrounding context in the image mirrors common crab environments like coral beds or crevices, reinforcing the classification.

### Detection Metrics:

- mAP@0.5 = 0.8905
- Precision & Recall > 0.85 across classes
- Classes: fish, crab, starfish, jellyfish, sea cucumber, sea anemone

#### Explanations:

- 3211 explanations generated across datasets
- Average explanation length: ~46 tokens
- Explanations grounded with top-3 BM25 knowledge snippets

### Futurework:
1. Expanding Knowledge Base
2. Building a Platform for inferencing

### Acknowledgement

This project was developed as part of the USRF-2025 Fellowship.
- Pushkar Sharma, B.Sc., IIT Patna
- Ayush Malviya, IIT Madras

Under mentorship of:
1. Dr. Gopendra Vikram Singh (Amity University, ACAI)
2. Dr. Alok Tiwari (Amity University, ACAI)
3. Rakesh Thakur(IISC researcher, Amity University ACAI)

Reference:
DU-MobileYOLO (Base Architecture):
Wang, X. et al. “DU-MobileYOLO: A Lightweight and Efficient Object Detector with Deformable Upsampling and Mobile Backbone.” IEEE Transactions on Artificial Intelligence, 2024.
