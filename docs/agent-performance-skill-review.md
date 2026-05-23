# TAO Skill Bank Agent Performance Review

Generated: 2026-05-22

## Scope

Reviewed `~/tao-skills-external` against Codex skill best practices and the
repo's own authoring rules. I inspected all `SKILL.md` files, plugin manifests,
helper scripts, references, templates, and validators.

Baseline checks run:

- `./scripts/validate-skills.sh`: passed, with 3 warnings in
  `data/deft-vlm-bcq-gap-analysis/SKILL.md`.
- `python3 -m py_compile` over packaged Python helper scripts: passed.
- `scripts/list_tao_capabilities.py`, `scripts/list_tao_models.py`, and
  `scripts/list_tao_platforms.py`: produced deterministic capability output.
- `scripts/verify-standalone.sh`: not run; it requires GPU Docker, NGC login,
  and potentially side-effecting container checks.

Pre-existing untracked files were present before review:

- `applications/tao-automl/eval-docker.config`
- `applications/tao-automl/eval.config`
- `applications/tao-automl/eval.slow-manual.config`
- `applications/tao-automl/scripts/`

## Executive Summary

The skill bank is strong in the places that matter most for hallucination
control: it has deterministic helper scripts, generated schema manifests,
container image resolution through `versions.yaml`, explicit preflight gates,
and a CI validator. Those are the right foundations.

The main improvement opportunity is not correctness of individual facts; it is
how much discretion the agent still has before it reaches those facts. Several
skills are too large, trigger metadata is uneven, some capabilities are present
on disk but missing from plugin registration, references are often long without
navigation, and some skills still contain harness-specific tool names despite
the repo's harness-agnostic goal.

Priority order:

1. Fix registration/discovery gaps so every production skill is either exposed
   or explicitly marked private.
2. Tighten frontmatter descriptions because they are the first trigger surface.
3. Refactor oversized `SKILL.md` bodies into lean routers plus references.
4. Add validation rules for the quality issues the current validator misses.
5. Convert repeated inline recipes into executable scripts with tests.

## Key Findings

### 1. Plugin Discovery Is Inconsistent

Evidence:

- `.codex-plugin/plugin.json` exposes only `./skills/`, which currently contains
  two router/discovery skills.
- `.claude-plugin/marketplace.json` exposes 74 direct skills, but there are 81
  non-template, non-example skill directories on disk.
- The following production skill directories are not listed in the main Claude
  `tao-skills` plugin:
  - `data/deft-vlm-bcq-gap-analysis`
  - `data/image-grounding`
  - `data/image-referring-expression`
  - `models/cosmos-embed`
  - `models/depth-net-fast-stereo/deploy`
  - `skills/tao-skill-bank-capabilities`
  - `skills/tao-workflow-launch`

Impact:

- Agents can mention capabilities that helper scripts discover from disk, while
  the active plugin may not expose the matching skill directly.
- Codex depends heavily on the two router skills. That can work, but only if the
  router skills always delegate to deterministic helpers and never answer from
  memory.

Recommendations:

- Make one manifest the source of truth for production skills.
- Add a validator rule: every `SKILL.md` outside `templates/` and example output
  must be registered in the intended plugin or listed in an explicit
  `registration_ignore` file with a reason.
- For Codex, choose deliberately between:
  - exposing all direct skills, or
  - keeping only router skills but generating a tested route map from manifests.
- Add route tests for prompts such as "run image grounding", "fine-tune
  cosmos-embed", and "deploy depth-net-fast-stereo".

### 2. Several Skill Bodies Are Too Large

Best practice is to keep `SKILL.md` lean and under roughly 500 lines, moving
variant details into references loaded on demand.

Skills over 500 lines:

- `applications/tao-automl/SKILL.md`: 1131
- `applications/tao-hf-finetune/SKILL.md`: 739
- `applications/rca-changenet/SKILL.md`: 594
- `applications/workflow-deft-aoi-loop/SKILL.md`: 566
- `models/dino/SKILL.md`: 534
- `platform/slurm/SKILL.md`: 519
- `platform/tao-sdk/SKILL.md`: 551

Skills from 250 to 500 lines also worth slimming:

- `applications/automl-deft-pipeline/SKILL.md`
- `applications/tao-hf-integration/SKILL.md`
- `data/deft-aoi-mining/SKILL.md`
- `data/deft-aoi-rca-vcn/SKILL.md`
- `data/deft-aoi-routing-vcn/SKILL.md`
- `models/cosmos-embed/SKILL.md`
- `models/cosmos-rl/SKILL.md`
- depth-net model and deploy skills
- `models/visual-changenet/SKILL.md`
- `platform/kubernetes/SKILL.md`
- `platform/lepton/SKILL.md`
- `skills/tao-workflow-launch/SKILL.md`

Impact:

- Long bodies increase token load before the agent knows which branch matters.
- Agents are more likely to mix instructions from unrelated actions, examples,
  platforms, or failure modes.

Recommendations:

- Keep only the trigger summary, preflight gate, core decision tree, and
  reference navigation in `SKILL.md`.
- Move algorithm catalogs, long examples, failure playbooks, model-specific
  inference mappings, and platform-specific install details into direct
  reference files.
- Add a validator warning at 350 lines and failure at 500 lines, with an allow
  list for exceptional skills.

### 3. Large Reference Files Lack Navigation

Reference files over 100 lines should have a short table of contents or a clear
grep/navigation block. Large Markdown references currently lack this.

High-impact examples:

- `applications/tao-hf-finetune/references/cv-scripts.md`: 884 lines
- `applications/tao-hf-finetune/references/vlm-scripts.md`: 760 lines
- `applications/tao-hf-integration/references/phase-3-implementation.md`: 807
  lines
- `applications/tao-hf-integration/references/workflow-consistency.md`: 810
  lines
- `applications/workflow-deft-aoi-loop/references/REPORT_RENDERING.md`: 174
  lines
- `data/image-referring-expression/references/configuration.md`: 209 lines

Impact:

- Progressive disclosure is weakened because loading a reference still brings in
  a lot of unrelated material.
- Without a TOC or search hints, agents may skim and miss a required section.

Recommendations:

- Add a "Contents" block to every Markdown reference over 100 lines.
- Split the largest references by task or action:
  - HF fine-tune scripts: classification, detection, segmentation, depth, VLM,
    LLM.
  - HF integration: config, trainer, deploy, packaging, tests.
  - Workflow loop rendering: data extraction, template rendering, validation.
- For very large code templates, prefer executable scripts or assets over
  Markdown code blocks.

### 4. Trigger Metadata Is Uneven

Frontmatter descriptions are the main trigger surface. Many model skills have
factual descriptions but no explicit "Use when..." phrase or user-language
synonyms.

Examples without explicit trigger wording:

- `models/dino/SKILL.md`
- `models/depth-net-mono/SKILL.md`
- `models/depth-net-stereo/SKILL.md`
- `models/mask2former/SKILL.md`
- many smaller model skills such as `action-recognition`, `bevfusion`,
  `centerpose`, `ocrnet`, `segformer`, and `vila`
- `data/daft-convert/SKILL.md` and `data/daft-validate/SKILL.md` say what they
  run but could include more literal user phrases.

The body of `applications/workflow-deft-aoi-loop/SKILL.md` has a "When to Use"
section, but body text is loaded only after triggering. Trigger phrases belong
in frontmatter first.

Impact:

- Skills may under-trigger for natural prompts.
- Agents may route by memory or folder names instead of metadata.

Recommendations:

- Rewrite every description as:
  - one factual sentence,
  - one "Use when..." clause with 2-5 literal trigger phrases,
  - key aliases such as `network_arch`, CLI names, task names, and common
    abbreviations.
- Add a validator warning for descriptions lacking "Use when" or fewer than two
  literal trigger phrases.
- Move duplicate "when to use" body text into frontmatter, leaving a short body
  orientation only when useful.

### 5. Harness-Agnostic Goal Conflicts With Harness-Specific Terms

The repo docs say skills should work across Claude Code, Codex, Gemini CLI, and
other Agent Skills-compatible agents. Some skills still include harness-specific
terms:

- `allowed-tools: ... WebFetch` in HF skills, while Codex has different web
  tooling.
- `allowed-tools: ... Task` and instructions to spawn a `reporter` subagent in
  `workflow-deft-aoi-loop`.
- `run_script()` is described as a Claude Code plugin helper.
- Several instructions reference Claude transcripts or Claude plugin cache paths.

Impact:

- Agents may attempt nonexistent tools.
- Portable installs become brittle.
- Hallucination risk rises when an agent tries to translate a harness-specific
  instruction on the fly.

Recommendations:

- Express tool requirements generically in portable skills:
  - "use the available web browsing tool" instead of `WebFetch`
  - "use an available subagent mechanism; otherwise run inline" instead of
    assuming `Task`
  - "run `python path/to/script.py` unless the harness provides a script runner"
- Keep harness-specific notes in short compatibility references.
- Add a validator rule for known harness-only names outside approved sections.

### 6. Frontmatter Schema Is Not Canonical

The repo intentionally uses more frontmatter than Codex's minimal
`name`/`description` shape, which is fine for a multi-runtime skill bank. The
problem is duplication and inconsistency.

Concrete issue from validator:

- `data/deft-vlm-bcq-gap-analysis/SKILL.md` has top-level `version`, `author`,
  and `tools`, but no `metadata.author`, `metadata.version`, or
  `allowed-tools`.

Other examples:

- `applications/tao-hf-finetune/SKILL.md` duplicates `version`/`author`/`tools`
  at top level and again under `metadata`/`allowed-tools`.
- Similar older-field patterns appear in `tao-hf-integration`, `tao-inference`,
  `nvidia-gpu-setup`, and DAFT skills.

Impact:

- Different runtimes or generators may read different fields.
- Metadata can become stale in one location.

Recommendations:

- Pick one bank schema:
  - required: `name`, `description`, `license`
  - optional: `metadata.compatibility`, `metadata.author`,
    `metadata.version`, `metadata.tags`, `allowed-tools`
- Remove duplicate top-level `author`, `version`, and `tools` after migration.
- Make the current warnings hard failures once the migration is done.

### 7. Too Much Executable Logic Lives In Prose

Some skills describe deterministic data transformations with inline Python
recipes or long shell snippets. Best practice is to bundle scripts when the
operation is fragile, repeatable, or validation-heavy.

Examples:

- `data/deft-aoi-routing-vcn/SKILL.md` describes a deterministic parquet/CSV
  split and report writer. This should be a small `scripts/route_vcn_gaps.py`
  with a `--help`, sample fixture, and test.
- `applications/tao-hf-finetune/references/cv-scripts.md` and
  `vlm-scripts.md` contain large scaffold code blocks. These are better as
  assets or generator scripts, with "fetch live docs" placeholders represented
  as explicit script extension points.
- Model inference mapping tables repeated in many model skills could be
  generated into structured reference files instead of manually maintained
  prose.

Impact:

- Agents can make transcription mistakes.
- Inline code is harder to test.
- Repeated prose increases drift.

Recommendations:

- Move deterministic recipes into scripts.
- Add tiny fixtures for data skills where possible.
- Add validator checks:
  - Python scripts compile.
  - executable files have shebangs.
  - files with shebangs intended to be run directly are executable.
  - each script referenced in `SKILL.md` exists and supports `--help` when
    applicable.

### 8. Stale TODOs and Known Limitations Leak Into Runtime Behavior

Examples:

- `models/dino/SKILL.md` contains a runtime TODO about extending
  `_apply_data_sources()`.
- `applications/automl-deft-pipeline/SKILL.md` tells the agent to file an issue
  or PR if an older DEFT skill hard-stops.

Impact:

- Agents may treat unresolved internal implementation work as part of the user
  flow.
- Ambiguous TODOs can lead to invented fixes or unnecessary pauses.

Recommendations:

- Move TODOs into issue tracker links or `docs/maintenance.md`.
- In skill bodies, state only the current supported behavior and the exact
  fallback.
- Add a validator warning for `TODO`, `FIXME`, and "known limitation" in
  production `SKILL.md` files unless explicitly allow-listed.

### 9. Codex UI Metadata Is Missing Per Skill

Codex skill best practices recommend `agents/openai.yaml` alongside each skill
for human-facing display name, short description, and default prompts. This repo
has plugin-level Codex interface metadata, but no per-skill `agents/openai.yaml`
files.

Impact:

- Codex users see the bank-level router experience, not rich per-skill chips.
- If direct Codex skill exposure is added later, the display metadata will need
  to be generated in a large follow-up migration.

Recommendations:

- If Codex continues to expose only the two router skills, generate
  `agents/openai.yaml` for those two and make their prompts strongly helper-led.
- If Codex moves to direct skill exposure, generate per-skill `agents/openai.yaml`
  from the normalized frontmatter descriptions and validate it for staleness.

## Strengths To Preserve

- Deterministic helper scripts are excellent. `list_tao_capabilities.py`,
  `list_tao_models.py`, `list_tao_platforms.py`, and `resolve_tao_image.py`
  give agents something concrete to run instead of guessing.
- `versions.yaml` centralizes image resolution and reduces stale image
  hallucinations.
- `schemas.manifest.json`, `automl_support.json`, and per-model schemas create
  objective gates for action and AutoML support.
- The launch gate in `skills/tao-workflow-launch/SKILL.md` is the right pattern:
  no side-effecting artifacts before platform, image, credential, dataset, and
  compute checks pass.
- The repo-level validator is already useful and should become stricter rather
  than be replaced.

## Recommended Validator Additions

Add checks for:

1. Unregistered production skill directories.
2. Codex plugin exposure: if `.codex-plugin/plugin.json` uses `./skills/`, then
   assert the router skills can enumerate every production capability.
3. `SKILL.md` length thresholds.
4. Reference Markdown files over 100 lines without a contents/navigation block.
5. Descriptions missing explicit trigger phrases.
6. Duplicate top-level `author`/`version`/`tools` when `metadata` and
   `allowed-tools` are present.
7. Harness-specific tool names outside approved compatibility notes.
8. `TODO`/`FIXME` in production skill bodies.
9. Script hygiene: py_compile, shebang/executable mismatch, referenced script
   existence, optional `--help` smoke.
10. Helper-output consistency: capabilities output must not advertise skills
    missing from the installed plugin bundle.

## Suggested Refactor Plan

Phase 1: Discovery and metadata

- Fix marketplace/Codex registration gaps.
- Normalize frontmatter.
- Rewrite descriptions for explicit triggers and aliases.
- Add validator rules for registration and frontmatter.

Phase 2: Progressive disclosure

- Slim the seven skills over 500 lines.
- Move long examples, API walkthroughs, error playbooks, and mappings into
  references.
- Add TOCs or split large Markdown references.

Phase 3: Deterministic execution

- Convert repeated inline recipes into scripts.
- Add fixtures and lightweight script tests.
- Expand `validate-skills.sh` to run syntax and script-reference checks.

Phase 4: Forward tests

Run fresh-agent prompts for route quality and hallucination resistance:

- "What TAO models support AutoML?"
- "Run image grounding on these images and captions."
- "Fine-tune cosmos-embed with AutoML."
- "Deploy depth-net-fast-stereo."
- "Train DINO on my COCO data on SLURM."
- "Run the AOI workflow end-to-end with AutoML and DEFT."

Each test should assert that the agent uses helpers/manifests, asks only for
required missing inputs, does not invent container images or dataset schemas,
and stops before side effects until the configured launch gate passes.

## Highest-ROI Fixes

1. Add the missing production skills to the plugin manifest or explicitly mark
   them private.
2. Tighten every model/data description with "Use when..." trigger phrases.
3. Split `applications/tao-automl/SKILL.md` and `applications/tao-hf-finetune/SKILL.md`.
4. Add validator rules for unregistered skills, long skills, long references,
   TODOs, and duplicate frontmatter fields.
5. Convert `data/deft-aoi-routing-vcn` into a tested script-driven skill.
