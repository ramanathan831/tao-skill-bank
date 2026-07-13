# DINOv3 method background — mapped to TAO config decisions

Read this before setting any non-default spec value. Every section ends in what to set and why.
Source of truth for implementation details: `nvidia_tao_pytorch/ssl/dinov3/README.md` in tao-pytorch.

## Lineage in one minute

- **DINO** (2021): self-distillation. A student ViT matches an EMA-averaged teacher's output
  distribution over multi-crop views (2 global + N local crops). Collapse is prevented by
  centering the teacher's outputs and sharpening with a low teacher temperature.
- **DINOv2** (2023): adds **iBOT** (masked-patch prediction — dense features), **KoLeo**
  (batch feature-spreading regularizer), Sinkhorn-Knopp centering as an option, and scale.
- **DINOv3** (Meta, 2025): patch-16 ViTs with **2D axial RoPE** instead of learned positional
  embeddings, **4 register tokens**, **Gram anchoring** to stop dense-feature degradation over
  long schedules, trained on LVD-1689M with a 7B flagship distilled down to S/S+/B/L/H+.
- **TAO `dinov3`**: *continual pre-training* of the public DINOv3 checkpoints on your unlabeled
  domain images — not from-scratch SSL. The whole recipe philosophy follows from that: the
  pretrained features are an asset to protect while nudging them toward the domain.

## Loss anatomy — what each term protects

`loss = DINO(CLS, global+local) + iBOT(masked patches) + 0.1 * KoLeo + [w_gram * Gram]`

| Term | Operates on | Protects | Config levers |
|---|---|---|---|
| DINO | CLS token | global-task features (classification, retrieval, anomaly detection) | `teacher_temperature`, `num_prototypes`, head |
| iBOT | masked patch tokens | dense-task features (segmentation, detection, depth) | (fixed weight 1.0) |
| KoLeo | CLS batch geometry | feature spread / retrieval | (fixed weight 0.1) |
| Gram | patch-token Gram matrix | dense feature *geometry* vs a frozen reference | `model.gram.*` |

Therefore: if **dense-task** transfer (segmentation, detection, depth) regresses while
global-task metrics hold, the dense terms (iBOT/Gram, and training precision — see recipes
doc) are the suspects. If **global-task** transfer regresses, look at drift: LR too high,
EMA momentum too low, too many steps.

## Continual pre-training discipline

The teacher is the deliverable and the anchor. TAO's defaults encode "adapt gently":

- **EMA momentum `val_base: 0.9999`** (vs 0.994 for from-scratch nvdinov2). Higher = teacher
  drifts slower = stronger anchor to pretrained features. Lower it only if the domain gap is
  huge and you accept forgetting.
- **LR**: `5e-5 * sqrt(global_batch / 1024)` (the `${eval:}` expression in the template
  auto-scales). Do not import from-scratch DINOv2 LRs (~1e-3 equivalent) — they destroy the
  checkpoint in thousands of steps.
- **Warmup must fit the schedule.** The template's `warm_up_steps: 10000` assumes a long
  single-node run. Rescale to ~5-10% of total optimizer steps
  (total steps = num_epochs * images / global_batch). An 8-node industrial run used 1250.
- **`last_layer_learning_rate.freeze_steps`** keeps the DINO head's last layer frozen early —
  standard DINO stabilization; scale it with warmup.
- **`teacher_temperature`** warms 0.04 → 0.07 over `warm_up_steps` (sharp early = stable
  targets). Rescale its `warm_up_steps` with schedule length too.

## RoPE and resolution

DINOv3 has no learnable `pos_embed`; position enters by rotating Q/K of patch tokens inside
attention (CLS/registers get identity rotation). Coordinates are normalized to [-1, 1], so
changing `img_size` 256 → 512 → 768 needs **no interpolation or surgery** — high-res adaptation
is a pure config change. Two hard rules:

- `rope_theta: 100.0` is parity-critical with the timm reference weights. **Never tune.**
- All crop sizes (`global_crops_size`, `local_crops_size`, export dims) must be **multiples of
  patch 16** (112, 224, 256, 336, 512, 768...).

## Gram anchoring

Long SSL schedules degrade patch-level (dense) features even while CLS-level metrics improve.
Gram anchoring fixes this by MSE-matching the student's patch-token cosine Gram matrix
(`normalize(X) @ normalize(X).T` — scale-invariant, fp32) to that of a **frozen Gram teacher**.
The paper treats it as a long-schedule / high-res / large-model tool — **not** used for small
backbones at base resolution. Knobs (`model.gram`):

- `enable` + `w_gram` (paper-style weight: 2.0) + `start_step` (activate after N steps).
- `teacher_source: pretrained` = frozen snapshot of the loaded weights (max anchor);
  `ema` + `refresh_interval` = early-EMA teacher refreshed every N steps (follows adaptation,
  weaker anchor).
- `teacher_scale`: Gram teacher runs at this multiple of student resolution, then its grid is
  average-pooled back. Paper uses 2.0; at 768 that is 96x96 = 9216 tokens — start at 1.0 and
  raise only if memory allows.

## Sinkhorn vs softmax centering

DINOv3 centers teacher head outputs with Sinkhorn-Knopp (SwAV-style), TAO default
`model.centering_method: sinkhorn`. It is numerically safe here (head outputs are cosine
similarities in [-1,1]). **If training shows instability or representation collapse (loss
plunging to a constant, feature variance dying), suspect centering first and fall back to
`model.centering_method: softmax`.**

## Architectures and starting checkpoints

| arch | embed/depth/heads | FFN | params | official timm checkpoint |
|---|---|---|---|---|
| vit_s | 384/12/6 | MLP | 21M | `vit_small_patch16_dinov3.lvd1689m` |
| vit_s_plus | 384/12/6 | SwiGLU | 29M | `vit_small_plus_patch16_dinov3.lvd1689m` |
| vit_b | 768/12/12 | MLP | 86M | `vit_base_patch16_dinov3.lvd1689m` |
| vit_l | 1024/24/16 | MLP | 300M | `vit_large_patch16_dinov3.lvd1689m` |
| vit_h_plus | 1280/32/20 | SwiGLU | 840M | `vit_huge_plus_patch16_dinov3.lvd1689m` |
| vit_7b | 4096/40/32 | SwiGLU(2.0) | 6.7B | `vit_7b_patch16_dinov3.lvd1689m` |

- Weights are **gated on Hugging Face** — the user must accept the `facebook/dinov3-*` license,
  then `hf download timm/<model>.lvd1689m --local-dir <weights_dir>`.
- `train.pretrained_model_path` accepts a directory (finds `model.safetensors` /
  `pytorch_model.bin`) or a direct `.safetensors`/`.pth`/`.bin` file. A validated key remapper
  converts timm layout to TAO layout (all tensors covered; only the iBOT `mask_token` stays
  freshly initialized).
- It also accepts **TAO DINOv3 checkpoints** (stripped `teacher_*.pth`/`student_*.pth` or full
  Lightning `.pth`) — that is how Phase 1 seeds from Phase 0.
- It does **not** accept DINOv2 / NVDINOv2 / arbitrary ViT weights.

## References

- DINOv3 (Meta AI, 2025) — Gram anchoring, RoPE, LVD-1689M family.
- DINOv2 (Meta AI, 2023) — iBOT + KoLeo + self-distillation recipe.
- DINO (Caron et al., 2021) — emerging properties in self-supervised ViTs.
- tao-pytorch `nvidia_tao_pytorch/ssl/dinov3/README.md` — implementation, checkpoint remapping,
  token layout, feature-parity guarantees (CLS/patch cosine > 0.99 vs timm).
