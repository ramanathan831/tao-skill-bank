# Maintenance — bumping container images and SDK wheel versions

All TAO container image tags and SDK wheel versions live in **one file**: [`versions.yaml`](../versions.yaml) at the repo root. RC bumps and image upgrades are a one-line edit there.

## Bumping a container image

Example: bumping the TAO Toolkit PyTorch image from `6.26.3` to `6.27.0`.

```diff
# versions.yaml
images:
  tao_toolkit:
-   pyt:        nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt
+   pyt:        nvcr.io/nvidia/tao/tao-toolkit:6.27.0-pyt
    cosmos_rl:  nvcr.io/nvidia/tao/tao-toolkit:6.26.3-cosmos-rl
    vila:       nvcr.io/nvidia/tao/tao-toolkit:6.26.3-vila
```

That's it. Every skill referencing `tao_toolkit.pyt` (28 of them today) automatically picks up the new tag at runtime.

### Verify the bump

```bash
./scripts/validate-skills.sh                        # confirms all key references still resolve
python -c "
from tao_sdk.versions import resolve_image
print(resolve_image('tao_toolkit.pyt'))
"   # expect the new tag
```

### Commit + PR

```bash
git add versions.yaml
git commit -m "Bump tao_toolkit.pyt to 6.27.0-pyt"
git push -u origin <your-branch>
```

CI runs `validate-skills.sh` automatically. Merge once green.

## Bumping an SDK wheel version

When `nvidia-tao-sdk` ships a new release:

```diff
# versions.yaml
wheels:
- tao_sdk:        nvidia-tao-sdk==0.2.0
- tao_sdk_lepton: nvidia-tao-sdk[lepton]==0.2.0
- tao_sdk_brev:   nvidia-tao-sdk[brev]==0.2.0
- tao_sdk_all:    nvidia-tao-sdk[all]==0.2.0
+ tao_sdk:        nvidia-tao-sdk==0.3.0
+ tao_sdk_lepton: nvidia-tao-sdk[lepton]==0.3.0
+ tao_sdk_brev:   nvidia-tao-sdk[brev]==0.3.0
+ tao_sdk_all:    nvidia-tao-sdk[all]==0.3.0
```

Skill preflight blocks reference `nvidia-tao-sdk[lepton]` (without a pinned version) so users get the latest from PyPI. The pinned versions in `versions.yaml` are for tooling that wants exact reproducibility (e.g., a CI image build).

## Adding a new image

1. Add an entry to `versions.yaml` under the appropriate group:

   ```yaml
   images:
     tao_toolkit:
       my_new_image: nvcr.io/nvidia/tao/tao-toolkit:6.26.3-my-new-image
   ```

2. In the skill's `references/skill_info.yaml`, reference by key:

   ```yaml
   container_image: tao_toolkit.my_new_image
   ```

3. Run the validator — confirms the key resolves.

## When to use absolute paths instead of keys

Both `container_image: tao_toolkit.pyt` (key) and `container_image: nvcr.io/.../tao-toolkit:6.26.3-pyt` (absolute) are valid indefinitely. Use absolute paths when:

- The image is **experimental** and not worth promoting to the manifest.
- The image is **third-party** (non-NVIDIA registry).
- The image is used by **only one skill** and unlikely to need a coordinated bump.

Promote to a key (`versions.yaml` entry) when:

- The image is shared by **two or more skills**.
- The image will be **bumped on a release cadence**.
- You want to track it in changelogs / RC notes.

## Related: Python wheel install matrix

Users install the SDK via:

```bash
pip install nvidia-tao-sdk            # core only
pip install nvidia-tao-sdk[lepton]    # + Lepton handler deps
pip install nvidia-tao-sdk[brev]      # + Brev handler (no extra Python deps)
pip install nvidia-tao-sdk[all]       # both extras
```

Legacy `tao-sdk` package: still installable as a thin alias that pulls in `nvidia-tao-sdk`. Prints a `DeprecationWarning` on import. Will be removed in a future major release.
