# SDA VCN (Visual ChangeNet) Workflow

Mining-based iterative data augmentation for binary image classification. Unlike the Cosmos SDA pipeline (which generates synthetic videos), VCN SDA mines similar images from a large source pool using embedding-based k-NN search, merges them with existing training data, and retrains.

## Use Case

PCB defect detection (Area of Interest / AOI). Visual ChangeNet compares image pairs (baseline vs. test) to classify defects as PASS or NO_PASS. The SDA loop identifies failure cases and augments the training set with similar images from an unlabeled source pool.

## Pipeline Stages

### Init Phase
1. **Zero-shot inference** — run VCN inference on KPI dataset with pretrained checkpoint
2. **Threshold optimization** — sweep siamese_score thresholds to maximize F1 while maintaining min_recall
3. **Zero-shot evaluation** — compute accuracy, precision, recall, F1 at optimal threshold
4. **SFT training** (conditional) — fine-tune on provided training data
5. **SFT inference + threshold + eval** (conditional) — evaluate the fine-tuned model

### Iteration Loop
1. **Gap analysis** — identify FP/FN from previous inference using optimal threshold
2. **Source pool expansion** — prepare gap queries and source pool for embedding
3. **Embed target** — SigLIP embeddings of gap/failure images
4. **Embed source** — SigLIP embeddings of source pool images
5. **k-NN mining** — find top-N nearest neighbors per gap image
6. **Merge training CSV** — combine mined samples with previous training data
7. **Training** — retrain VCN on merged dataset
8. **Inference** — run inference on KPI set with new checkpoint
9. **Threshold optimization** — re-optimize threshold for new model
10. **Evaluation** — compute metrics at new threshold

## Data Format

**CSV annotations** (VCN format):
```csv
input_path,object_name,label
aoi_001,chip_A,PASS
aoi_002,chip_B,NO_PASS
```

**Lighting conditions**: Images are stored as `{images_dir}/{input_path}/{object_name}_{lighting}{ext}` where lighting comes from the config's `input_map` (e.g., SolderLight, UniformLight).

## Prerequisites

- **kpi_dataset_uri**: S3 URI containing KPI annotations CSV and images/ folder
- **mining_source_csv**: S3 URI of source pool CSV (large dataset to mine from)
- **mining_source_images**: S3 URI of source pool images directory
- **train_dataset_uri** (optional): Initial training data for SFT before SDA loop
- **init_checkpoint** (optional): Pretrained VCN checkpoint

## Key Parameters

- **topn**: Number of nearest neighbors per gap query (default: 5)
- **knn_metric**: Distance metric for k-NN (cosine or euclidean, default: cosine)
- **min_recall**: Minimum NO_PASS recall threshold must achieve (default: 1.0)
- **continual_model**: Chain checkpoints across iterations (default: true)
- **continual_dataset**: Accumulate training data across iterations (default: true)

## Storage Layout

```
{storage_root}/
  init/
    inference/         # Zero-shot inference results
    threshold.json     # Optimal threshold
    metrics.json       # Eval metrics
  iter_1/
    gaps/gaps.parquet  # Failure cases
    embeddings/        # SigLIP embeddings
    mining/            # k-NN mined pairs
    training_data/     # Merged training CSV
    inference/         # Iteration inference results
    threshold.json     # Iteration threshold
    eval_metrics.json  # Iteration metrics
```
