# Enhanced-Atrial-Fibrillation-Prediction-in-ESUS-Patients-with-Pre-training-and-Transfer-Learning

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

### Workflow Summary
1. **Generate features** → `gen_feat.sh`  
2. **Compute overlapness & homogeneity** → `gen_overlap_homogeneity.sh`  
3. **Train low-dimensional embeddings** → run one of `spicd3.sh`, `spicd4.sh`, `cbicd3.sh`, or `cbicd4.sh` depending on dataset.  

All scripts are located under the `scripts/` folder and can be executed directly in sequence.
