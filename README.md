# Enhanced-Atrial-Fibrillation-Prediction-in-ESUS-Patients-with-Pre-training-and-Transfer-Learning
**Keywords:** Atrial fibrillation, ESUS, Hypergraph learning, Pre-training, Transfer learning

## Project Summary
This repo explores **Atrial Fibrillation (AF) prediction** in **ESUS** patients using **pre-training + transfer learning** on hypergraphs. We first learn patient representations on a large stroke cohort (AI-RESPECT, *n* = 7,780), then transfer compact embeddings to a smaller ESUS cohort (*n* = 510) for downstream AF prediction with lightweight ML models.

---

## Task Description
**Goal:** Predict whether an ESUS patient will develop AF (binary classification).

- **Challenges:** small target cohort, high-dimensional diagnostic features (ICD), risk of overfitting.
- **Inputs:** 
  - Baseline clinical features (58 dims)
  - Diagnostic features (ICD; up to 1,529 dims for ESUS)
- **Output:** AF risk (0/1)

---

## Method Overview
We represent patient data as a **hypergraph**: nodes = diagnostic features; hyperedges = patient visits/encounters. We then learn **hyperedge (patient) embeddings** and transfer them to ESUS.

### 1) From-Scratch (Baseline)
Concatenate clinical + diagnostic features:
- `x_i = x_{i,b} ⊕ x_{i,d}`
- Trains directly on ESUS; simple but prone to overfitting in small-N, high-D.

### 2) Supervised Transfer
Pre-train a **hypergraph transformer** on AI-RESPECT (labeled PSCI task) and transfer:
- Learn final hyperedge embedding `x^L_{e,i}` as a **32-D patient vector**.
- Build ESUS features by `x_i = x_{i,tr} ⊕ x_{i,b}` and train AF classifiers (LR/RF/GB).
- Benefits: injects structure + priors from a large related cohort.

### 3) Unsupervised Transfer
Pre-train on AI-RESPECT **without labels** via two components, then transfer:
- **Hypergraph View Augmentation (genSim):** 
  - Node masking biased by duplication; hyperedge selection via **Gumbel-Softmax**.
  - Consistency objectives across two augmented views: `L_genSim = L_hyper + L_sim`.
- **Triplet Contrastive Learning (Trip):**
  - **Node-level**, **hyperedge-level**, and **membership-level** contrasts across augmented graphs.
  - Total loss: `L_total = L_genSim + L_n + L_e + L_m` (equal weights).
- Extract a **32-D** patient embedding `x_{i,tr}` and concatenate with clinical features as above.

---

## Why Hypergraphs?
Hypergraphs naturally capture **many-to-many** relations between features and visits, enabling attention-based message passing:
- Within-hyperedge (V→E) and within-node (E→V) self-attention propagate information between **features** and **patient encounters**.
- Produces compact, expressive **patient embeddings** for downstream AF prediction.

---

## Usage

### Unsupervised Representation Learning for Hyperedges

#### Step 1: Feature Generation
Navigate to the `scripts/` folder to generate initial hyperedge features:
- **Script**: `gen_feat.sh`
- **Description**: Runs random walks + Word2Vec to produce node embeddings and hyperedge feature representations.

#### Step 2: Overlapness and Homogeneity
After generating features, compute overlapness and homogeneity metrics:
- **Script**: `gen_overlap_homogeneity.sh`
- **Description**: Generates `overlapness` and `homogeneity` files for each dataset.
- **Note**: Place the generated files under the same folder as the corresponding dataset.

#### Step 3: Low-Dimensional Representation
Once the overlapness and homogeneity files are ready, run the training scripts to obtain low-dimensional hyperedge feature representations:
- **Scripts**:
  - `spicd3.sh` – train on **separate ICD-3** dataset
  - `spicd4.sh` – train on **separate ICD-4** dataset
  - `cbicd3.sh` – train on **combined ICD-3** dataset
  - `cbicd4.sh` – train on **combined ICD-4** dataset
- **Description**: Each script calls the transfer learning framework to pretrain and fine-tune on hypergraph datasets, producing compressed representations of hyperedge features.

---

