#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Validate the skill bank.
#
# Required:
#   1. Every skill path in .claude-plugin/marketplace.json and .codex-plugin/plugin.json
#      resolves to a dir with SKILL.md.
#   2. The Codex-facing skills/ directory has no symlink mirror of canonical skills.
#   3. Every SKILL.md has valid YAML frontmatter with `name` and `description`.
#   4. Each SKILL.md body contains enough info to run the skill (heuristic: a Quick Start
#      section, a docker run code block, OR a references/skill_info.yaml link).
#   5. No SDK symbols leak into model/data/application SKILL.md (platform/* exempt).
#   6. Hook paths in skill frontmatter resolve to existing scripts.
#
# Optional (validated only if the file exists):
#   7. references/skill_info.yaml parses; if the skill is in models/ or data/ and declares
#      it, container_image + at least one actions.*.command must be present.
#   8. references/model_info.yaml (legacy name) parses if present — same rules.
#
# Exit status = number of errors found.
#
# Usage:
#   ./scripts/validate-skills.sh                  # full validation
#   ./scripts/validate-skills.sh --quick          # skip optional structured-metadata checks

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MARKETPLACE=".claude-plugin/marketplace.json"
CODEX_PLUGIN=".codex-plugin/plugin.json"
errors=0

ok()  { echo "  OK: $*"; }

# ─── 1. marketplace/plugin paths ─────────────────────────────────────────────
echo "=== 1. marketplace/plugin skill paths ==="
[ -f "$MARKETPLACE" ] || { echo "ERROR: $MARKETPLACE not found"; exit 1; }
[ -f "$CODEX_PLUGIN" ] || { echo "ERROR: $CODEX_PLUGIN not found"; exit 1; }

python3 - <<'PY'
import json, os, sys

errs = 0

def check_skill_path(owner, path):
    global errs
    real = path.lstrip('./')
    skill_md = os.path.join(real, 'SKILL.md')
    if not os.path.isfile(skill_md):
        print(f"ERROR: {owner} → {path} (no SKILL.md at {skill_md})", file=sys.stderr)
        errs += 1

with open('.claude-plugin/marketplace.json') as f:
    mp = json.load(f)
for plugin in mp.get('plugins', []):
    for path in plugin.get('skills', []):
        check_skill_path(plugin['name'], path)

with open('.codex-plugin/plugin.json') as f:
    cp = json.load(f)
skills = cp.get('skills', [])
if isinstance(skills, str):
    skills = [skills]
elif not isinstance(skills, list):
    print("ERROR: .codex-plugin/plugin.json → skills must be a string or list", file=sys.stderr)
    errs += 1
    skills = []
for path in skills:
    check_skill_path('codex-plugin', path)

sys.exit(errs)
PY
[ $? -eq 0 ] && ok "all marketplace/plugin paths resolve" || errors=$((errors + $?))

# ─── 1b. production skills are registered ───────────────────────────────────
echo
echo "=== 1b. production SKILL.md registration ==="
if python3 - <<'PY'
import json, os, sys
from pathlib import Path

def plugin_paths(path, plugin_name=None):
    with open(path) as f:
        data = json.load(f)
    if 'plugins' in data:
        paths = set()
        for plugin in data.get('plugins', []):
            if plugin_name is None or plugin.get('name') == plugin_name:
                paths.update(p.rstrip('/') for p in plugin.get('skills', []))
        return paths
    skills = data.get('skills', [])
    if isinstance(skills, str):
        skills = [skills]
    return {p.rstrip('/') for p in skills}

def scoped_out(path):
    return (
        path.startswith('./applications/')
        and path != './applications/tao-automl'
    ) or path == './skills/tao-workflow-launch'

claude_registered = plugin_paths('.claude-plugin/marketplace.json', 'tao-skills')
codex_registered = plugin_paths('.codex-plugin/plugin.json')

production = set()
for skill_md in Path('.').rglob('SKILL.md'):
    parts = skill_md.parts
    if '.git' in parts or 'plugins' in parts or 'templates' in parts or 'examples' in parts:
        continue
    production.add('./' + str(skill_md.parent))

claude_missing = sorted(production - claude_registered)
if claude_missing:
    for path in claude_missing:
        print(f"ERROR: production skill is not registered in tao-skills: {path}", file=sys.stderr)
codex_required = {path for path in production if not scoped_out(path)}
codex_missing = sorted(codex_required - codex_registered)
if codex_missing:
    for path in codex_missing:
        print(f"ERROR: in-scope production skill is not registered in codex plugin: {path}", file=sys.stderr)
if claude_missing or codex_missing:
    sys.exit(1)
sys.exit(0)
PY
then
  ok "all production skills registered in tao-skills and in-scope skills registered in codex plugin"
else
  errors=$((errors + 1))
fi

# ─── 1b. Codex skills/ should not mirror canonical skills ──────────────────
echo
echo "=== 1b. Codex skills/ has no mirror symlinks ==="
codex_skill_symlinks="$(find skills -mindepth 1 -maxdepth 1 -type l -print | sort || true)"
if [ -n "$codex_skill_symlinks" ]; then
  codex_skill_symlink_errors=0
  while IFS= read -r path; do
    [ -z "$path" ] && continue
    echo "ERROR: $path — do not mirror canonical skills under skills/. Keep real skills under applications/, data/, models/, or platform/." >&2
    codex_skill_symlink_errors=$((codex_skill_symlink_errors + 1))
  done <<< "$codex_skill_symlinks"
  errors=$((errors + codex_skill_symlink_errors))
else
  ok "skills/ contains only Codex helper skill directories"
fi

# ─── 2. SKILL.md frontmatter (errors) + DAFT-style optional fields (warnings) ─
echo
echo "=== 2. SKILL.md frontmatter ==="
python3 - <<'PY'
import os, sys, yaml, re
errs = 0
warns = 0
allowed_top_level = {'name', 'description', 'license', 'metadata', 'allowed-tools'}

def scoped_out(skill_md):
    return (
        skill_md.startswith('applications/')
        and skill_md != 'applications/tao-automl/SKILL.md'
    ) or skill_md == 'skills/tao-workflow-launch/SKILL.md'

def warn(msg):
    global warns
    print(f"WARN: {msg}", file=sys.stderr)
    warns += 1

def error(msg):
    global errs
    print(f"ERROR: {msg}", file=sys.stderr)
    errs += 1

def iter_skill_files():
    for root, dirs, files in os.walk('.', followlinks=False):
        dirs[:] = [
            d for d in dirs
            if d not in ('.git', 'plugins')
            and 'templates/skill-skeleton' not in os.path.join(root, d)
            and not os.path.islink(os.path.join(root, d))
        ]
        if 'SKILL.md' in files:
            yield os.path.join(root, 'SKILL.md').lstrip('./')

for skill_md in iter_skill_files():
    with open(skill_md) as f:
        content = f.read()
    m = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    if not m:
        error(f"{skill_md} — missing or malformed frontmatter"); continue
    try:
        fm = yaml.safe_load(m.group(1))
    except yaml.YAMLError as e:
        error(f"{skill_md} — YAML parse error: {e}"); continue
    if not isinstance(fm, dict):
        error(f"{skill_md} — frontmatter is not a mapping"); continue
    unexpected = sorted(set(fm) - allowed_top_level)
    if unexpected:
        msg = f"{skill_md} — unexpected top-level frontmatter keys: {', '.join(unexpected)}; move runtime/catalog data under metadata"
        if scoped_out(skill_md):
            warn(msg)
        else:
            error(msg)
    # Required fields
    if 'name' not in fm:
        error(f"{skill_md} — missing `name`")
    if 'license' not in fm:
        error(f"{skill_md} — missing `license`. Add `license: Apache-2.0` (see docs/authoring.md).")
    # Optional fields — warn but don't fail
    metadata = fm.get('metadata') if isinstance(fm.get('metadata'), dict) else {}
    if 'compatibility' not in metadata and 'compatibility' not in fm:
        warn(f"{skill_md} — missing `metadata.compatibility` (runtime requirements). See docs/authoring.md for examples.")
    if 'author' not in metadata:
        warn(f"{skill_md} — missing `metadata.author`. Add `author: NVIDIA Corporation`.")
    elif metadata.get('author') != 'NVIDIA Corporation':
        error(f"{skill_md} — `metadata.author` must be exactly 'NVIDIA Corporation' (found: {metadata.get('author')!r}).")
    if 'version' not in metadata:
        warn(f"{skill_md} — missing `metadata.version`. Add e.g. `version: \"0.1\"`.")
    if 'allowed-tools' not in fm:
        warn(f"{skill_md} — missing `allowed-tools`. Set if the skill uses Read/Bash/Write frequently.")
    if 'description' not in fm:
        error(f"{skill_md} — missing `description`")
if warns > 0:
    print(f"  ({warns} warning(s) — see docs/authoring.md to address)", file=sys.stderr)
sys.exit(errs)
PY
[ $? -eq 0 ] && ok "all SKILL.md frontmatter valid" || errors=$((errors + $?))

# ─── 2b. quality checks for agent performance ────────────────────────────────
echo
echo "=== 2b. agent-performance quality checks ==="
if python3 - <<'PY'
import os, re, sys, yaml
from pathlib import Path

warns = 0
errs = 0

def scoped_out(path):
    text = str(path)
    return (
        text.startswith('applications/')
        and text != 'applications/tao-automl/SKILL.md'
    ) or text == 'skills/tao-workflow-launch/SKILL.md'

def report(path, msg):
    global warns
    global errs
    if scoped_out(path):
        print(f"WARN: {path} — {msg}", file=sys.stderr)
        warns += 1
    else:
        print(f"ERROR: {path} — {msg}", file=sys.stderr)
        errs += 1

def iter_skill_files():
    for path in Path('.').rglob('SKILL.md'):
        parts = path.parts
        if '.git' in parts or 'plugins' in parts or 'templates' in parts or 'examples' in parts:
            continue
        yield path

for path in iter_skill_files():
    content = path.read_text()
    lines = content.splitlines()
    if len(lines) > 500:
        report(path, f"{len(lines)} lines; prefer <500 lines and move details to references/")
    if re.search(r'\b(TODO|FIXME)\b|known limitation', content, re.IGNORECASE):
        report(path, "contains TODO/FIXME/known limitation language; move maintenance notes out of runtime instructions")

    m = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    if not m:
        continue
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        continue
    desc = str(fm.get('description', ''))
    if not re.search(r'\buse when\b', desc, re.IGNORECASE):
        report(path, "description lacks explicit 'Use when' trigger language")
    legacy = [k for k in ('author', 'version', 'tools') if k in fm]
    if legacy:
        report(path, f"duplicate/legacy frontmatter fields present: {', '.join(legacy)}")
    top_level_catalog = [k for k in ('compatibility', 'tags') if k in fm]
    if top_level_catalog:
        report(path, f"catalog/runtime frontmatter should live under metadata: {', '.join(top_level_catalog)}")

for path in Path('.').rglob('references/*.md'):
    parts = path.parts
    if '.git' in parts or 'plugins' in parts or 'templates' in parts or 'examples' in parts:
        continue
    lines = path.read_text(errors='replace').splitlines()
    if len(lines) <= 100:
        continue
    head = '\n'.join(lines[:30]).lower()
    if not any(marker in head for marker in ('table of contents', '## contents', '## toc', 'search hints', 'navigation')):
        report(path, f"{len(lines)} lines and no Contents/search-hints block near the top")

if warns:
    print(f"  ({warns} scoped workflow/application quality warning(s) — non-blocking by current scope)", file=sys.stderr)
sys.exit(errs)
PY
then
  ok "quality checks passed for in-scope skills"
else
  errors=$((errors + 1))
fi

# ─── 2c. Codex UI metadata for directly exposed skills ───────────────────────
echo
echo "=== 2c. codex agents/openai.yaml metadata ==="
if python3 - <<'PY'
import json, re, sys, yaml
from pathlib import Path

errs = 0
with open('.codex-plugin/plugin.json') as f:
    plugin = json.load(f)

skills = plugin.get('skills', [])
if isinstance(skills, str):
    skills = [skills]

for skill_path in skills:
    root = Path(skill_path)
    skill_md = root / 'SKILL.md'
    try:
        skill_content = skill_md.read_text()
        match = re.match(r'^---\n(.*?)\n---', skill_content, re.DOTALL)
        skill_frontmatter = yaml.safe_load(match.group(1)) if match else {}
        skill_name = skill_frontmatter.get('name') if isinstance(skill_frontmatter, dict) else None
    except Exception as exc:
        print(f"ERROR: {skill_md} — unable to read frontmatter: {exc}", file=sys.stderr)
        errs += 1
        skill_name = None
    openai_yaml = root / 'agents' / 'openai.yaml'
    if not openai_yaml.is_file():
        print(f"ERROR: {skill_path} — missing agents/openai.yaml", file=sys.stderr)
        errs += 1
        continue
    try:
        data = yaml.safe_load(openai_yaml.read_text()) or {}
    except yaml.YAMLError as exc:
        print(f"ERROR: {openai_yaml} — YAML parse error: {exc}", file=sys.stderr)
        errs += 1
        continue
    interface = data.get('interface')
    if not isinstance(interface, dict):
        print(f"ERROR: {openai_yaml} — missing interface mapping", file=sys.stderr)
        errs += 1
        continue
    for key in ('display_name', 'short_description', 'default_prompt'):
        value = interface.get(key)
        if not isinstance(value, str) or not value.strip():
            print(f"ERROR: {openai_yaml} — missing interface.{key}", file=sys.stderr)
            errs += 1
    short = interface.get('short_description', '')
    if isinstance(short, str) and short and not (25 <= len(short) <= 64):
        print(f"ERROR: {openai_yaml} — interface.short_description must be 25-64 chars", file=sys.stderr)
        errs += 1
    default_prompt = interface.get('default_prompt', '')
    if isinstance(skill_name, str) and isinstance(default_prompt, str) and f'${skill_name}' not in default_prompt:
        print(f"ERROR: {openai_yaml} — interface.default_prompt must mention ${skill_name}", file=sys.stderr)
        errs += 1

sys.exit(errs)
PY
then
  ok "codex metadata present for directly exposed skills"
else
  errors=$((errors + 1))
fi

# ─── 3. SKILL.md body has runnable info ─────────────────────────────────────
echo
echo "=== 3. SKILL.md body has runnable info ==="
python3 - <<'PY'
import os, sys, re
# A SKILL.md is "runnable" if any of:
#   - body has a "## Quick Start" or "## Quick start" heading
#   - body has a `docker run` code block
#   - body has a Python `sdk.create_job` call (for SDK-driven skills)
#   - the skill dir has references/skill_info.yaml or references/model_info.yaml on disk
# Skips templates/.
errs = 0

def iter_skill_files():
    for root, dirs, files in os.walk('.', followlinks=False):
        dirs[:] = [
            d for d in dirs
            if d not in ('.git', 'plugins')
            and 'templates/skill-skeleton' not in os.path.join(root, d)
            and not os.path.islink(os.path.join(root, d))
        ]
        if 'SKILL.md' in files:
            yield os.path.join(root, 'SKILL.md').lstrip('./')

for skill_md in iter_skill_files():
    skill_dir = os.path.dirname(skill_md)
    with open(skill_md) as f:
        content = f.read()
    has_qs = re.search(r'^##\s+quick ?start', content, re.IGNORECASE | re.MULTILINE)
    has_dr = 'docker run' in content
    has_sdk = re.search(r'sdk\.create_job|LeptonSDK|BrevSDK', content)
    has_refs = (os.path.isfile(os.path.join(skill_dir, 'references/skill_info.yaml'))
                or os.path.isfile(os.path.join(skill_dir, 'references/model_info.yaml')))
    # Local-Python or agent-prompt-driven skills: presence of scripts/ or hooks/ counts as runnable.
    has_scripts = os.path.isdir(os.path.join(skill_dir, 'scripts'))
    has_hooks = os.path.isdir(os.path.join(skill_dir, 'hooks'))
    if not (has_qs or has_dr or has_sdk or has_refs or has_scripts or has_hooks):
        print(f"ERROR: {skill_md} — no runnable info found. Add a Quick Start, docker run block, SDK call, references/skill_info.yaml, scripts/, or hooks/.", file=sys.stderr)
        errs += 1
sys.exit(errs)
PY
[ $? -eq 0 ] && ok "all SKILL.md bodies have runnable info" || errors=$((errors + $?))

# ─── 4. no SDK leaks in model/data/application skills ───────────────────────
echo
echo "=== 4. no SDK leaks in model/data/application skills ==="
python3 - <<'PY'
import re, os, sys
leak_re = re.compile(r'tao_sdk|TaoExecutionSDK|sdk\.create_job|sdk\.list_path|sdk\.check_path|execute_step|agent_runner|script_runner')
errs = 0
for root, dirs, files in os.walk('.'):
    if any(x in root for x in ('.git', 'templates/skill-skeleton')):
        continue
    if 'SKILL.md' in files:
        path = os.path.join(root, 'SKILL.md')
        # Platform skills legitimately document the SDK
        if path.startswith('./platform/'):
            continue
        # Application skills that are SDK-orchestrated (AutoML, etc.) are exempt.
        # Add new ones here only after confirming they cannot run without the SDK.
        if path in ('./applications/tao-automl/SKILL.md',):
            continue
        # Models may have an "Optional: running via the TAO SDK" section
        is_model = path.startswith('./models/')
        with open(path) as f:
            content = f.read()
        matches = leak_re.findall(content)
        if not matches:
            continue
        if is_model:
            opt = re.search(r'##\s*Optional:.*?(?=\n##\s|\Z)', content, re.DOTALL | re.IGNORECASE)
            if opt:
                outside = leak_re.findall(content.replace(opt.group(0), ''))
                if outside:
                    print(f"ERROR: {path} — SDK symbols outside Optional SDK section: {outside[:3]}", file=sys.stderr); errs += 1
                continue
            print(f"ERROR: {path} — SDK symbols found: {matches[:3]}. Wrap in an 'Optional: running via the TAO SDK' section or remove.", file=sys.stderr); errs += 1
        else:
            print(f"ERROR: {path} — SDK symbols in non-model skill: {matches[:3]}", file=sys.stderr); errs += 1
sys.exit(errs)
PY
[ $? -eq 0 ] && ok "no SDK symbol leaks" || errors=$((errors + $?))

# ─── 5. hook paths resolve ──────────────────────────────────────────────────
echo
echo "=== 5. hook paths resolve ==="
python3 - <<'PY'
import re, os, sys, yaml
errs = 0
for root, dirs, files in os.walk('.'):
    if any(x in root for x in ('.git', 'templates/skill-skeleton')):
        continue
    if 'SKILL.md' not in files: continue
    path = os.path.join(root, 'SKILL.md')
    with open(path) as f:
        content = f.read()
    m = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    if not m: continue
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError: continue
    hooks = fm.get('hooks') or {}
    if not isinstance(hooks, dict): continue
    for event, entries in hooks.items():
        for entry in (entries or []):
            for hook in (entry.get('hooks') or []):
                cmd = hook.get('command', '')
                for m2 in re.finditer(r'\$\{CLAUDE_SKILL_DIR\}/([^\s"\']+)', cmd):
                    rel = m2.group(1)
                    full = os.path.join(root, rel)
                    if not os.path.exists(full):
                        print(f"ERROR: {path} — hook references missing file: {rel}", file=sys.stderr); errs += 1
sys.exit(errs)
PY
[ $? -eq 0 ] && ok "all hook paths resolve" || errors=$((errors + $?))

# ─── 5b. script hygiene ─────────────────────────────────────────────────────
echo
echo "=== 5b. script hygiene ==="
if python3 - <<'PY'
import py_compile, stat, sys, tempfile
from pathlib import Path

errs = 0
warns = 0

def scoped_out(path):
    text = str(path)
    return (
        text.startswith('applications/')
        and text != 'applications/tao-automl/SKILL.md'
    ) or text.startswith('skills/tao-workflow-launch/')

def hygiene_warn(path, msg):
    global warns
    global errs
    if scoped_out(path):
        print(f"WARN: {path} — {msg}", file=sys.stderr)
        warns += 1
    else:
        print(f"ERROR: {path} — {msg}", file=sys.stderr)
        errs += 1

script_files = []
python_files = []
for path in Path('.').rglob('*'):
    if not path.is_file():
        continue
    parts = path.parts
    if '.git' in parts or 'plugins' in parts or 'templates' in parts or 'examples' in parts:
        continue
    if 'scripts' in parts:
        script_files.append(path)
    if path.suffix == '.py' and ('scripts' in parts or 'references' in parts):
        python_files.append(path)

with tempfile.TemporaryDirectory(prefix='tao-skill-pycompile-') as pycache_tmp:
    pycache_root = Path(pycache_tmp)
    for path in python_files:
        try:
            cfile = pycache_root / (str(path).replace('/', '__') + '.pyc')
            py_compile.compile(str(path), cfile=str(cfile), doraise=True)
        except py_compile.PyCompileError as e:
            print(f"ERROR: {path} — Python syntax check failed: {e.msg}", file=sys.stderr)
            errs += 1

for path in script_files:
    text = path.read_text(errors='ignore')
    executable = bool(path.stat().st_mode & stat.S_IXUSR)
    has_shebang = text.startswith('#!')
    if executable and not has_shebang:
        hygiene_warn(path, "executable bit set but no shebang")
    if has_shebang and not executable:
        hygiene_warn(path, "has shebang but is not executable")

if warns:
    print(f"  ({warns} script hygiene warning(s) — non-blocking)", file=sys.stderr)
sys.exit(errs)
PY
then
  ok "script hygiene checks passed"
else
  errors=$((errors + 1))
fi

# ─── 6/7. optional structured metadata ──────────────────────────────────────
if [ "${1:-}" != "--quick" ]; then
  echo
  echo "=== 6. references/skill_info.yaml + legacy model_info.yaml (when present) ==="
  python3 - <<'PY'
import os, sys, yaml
errs = 0
for root, dirs, files in os.walk('.'):
    if any(x in root for x in ('.git', 'templates/skill-skeleton', '.claude-plugin')):
        continue
    if 'references' not in root: continue
    for fname in ('skill_info.yaml', 'model_info.yaml'):
        if fname in files:
            path = os.path.join(root, fname)
            try:
                with open(path) as f:
                    info = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(f"ERROR: {path} — YAML parse error: {e}", file=sys.stderr); errs += 1; continue
            skill_dir = os.path.dirname(os.path.dirname(path))
            # Validate container_image: must be either a key reference resolving in
            # versions.yaml, or an absolute registry URI.
            if isinstance(info, dict) and isinstance(info.get('container_image'), str):
                img = info['container_image']
                # Absolute path heuristic: contains '/' or ':' (registry URI shape).
                if '/' in img or ':' in img:
                    pass  # accept as literal — both forms valid
                else:
                    # Treat as key reference; resolve against versions.yaml
                    try:
                        with open('versions.yaml') as vf:
                            manifest = yaml.safe_load(vf) or {}
                        node = manifest.get('images', {})
                        for part in img.split('.'):
                            if not isinstance(node, dict) or part not in node:
                                raise KeyError(f"key '{img}' missing from versions.yaml images tree")
                            node = node[part]
                        if not isinstance(node, str):
                            print(f"ERROR: {path} — container_image key '{img}' resolves to non-string in versions.yaml", file=sys.stderr); errs += 1
                    except FileNotFoundError:
                        print(f"ERROR: {path} — container_image '{img}' looks like a key but versions.yaml is missing at repo root", file=sys.stderr); errs += 1
                    except KeyError as e:
                        print(f"ERROR: {path} — container_image key '{img}' not found in versions.yaml ({e})", file=sys.stderr); errs += 1
            # If this is a model or data skill AND skill_info declares actions, validate them
            if (skill_dir.startswith('./models/') or skill_dir.startswith('./data/')) and isinstance(info, dict):
                if 'actions' in info and not info.get('container_image'):
                    print(f"WARN: {path} — has actions but no container_image", file=sys.stderr)
                actions = info.get('actions') or {}
                for name, spec in actions.items():
                    if 'command' not in (spec or {}):
                        print(f"ERROR: {path} — actions.{name} missing `command`", file=sys.stderr); errs += 1
sys.exit(errs)
PY
  [ $? -eq 0 ] && ok "skill_info.yaml / model_info.yaml validation passed" || errors=$((errors + $?))
fi

echo
if [ $errors -eq 0 ]; then
  echo "✓ validate-skills passed"
  exit 0
else
  echo "✗ validate-skills failed: $errors error(s)"
  exit $errors
fi
