#!/usr/bin/env bash
# PreCompact hook: if a DEFT loop is active, inject a reminder for Claude
# to checkpoint any in-flight state to deft_state.json before compaction.

set -euo pipefail

WORKSPACE="${DEFT_WORKSPACE:-$PWD}"

STATE_FILE=$(find "$WORKSPACE" -name "deft_state.json" -not -path "*/node_modules/*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$STATE_FILE" ]; then
  exit 0
fi

RESULTS_DIR=$(dirname "$STATE_FILE")

# Read current step from state file
CURRENT_STEP=$(python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
print(f\"iter={state.get('current_iteration', '?')}, step={state.get('current_step', '?')}\")
" 2>/dev/null || echo "unknown")

python3 -c "
import json, sys
context = '''DEFT LOOP ACTIVE — COMPACTION IMMINENT.

Before context is compacted, you MUST checkpoint your current progress:

1. Update ${RESULTS_DIR}/deft_state.json with:
   - current_iteration and current_step reflecting where you are RIGHT NOW
   - Any metrics, paths, or counts from steps completed since the last checkpoint
   Last recorded state: ${CURRENT_STEP}

2. If you are mid-iteration and have RCA findings, strategy decisions, or
   partial results not yet written to disk, write them now:
   - Iteration summary: \${RESULTS_DIR}/iterN_summary.md
   - Any key decision (arm selection, target defects) into the state file

3. After checkpointing, compaction can proceed safely — the PostCompact hook
   will restore state + summaries into your refreshed context.

Do this NOW before responding to anything else.'''

output = {
    'hookSpecificOutput': {
        'hookEventName': 'PreCompact',
        'additionalContext': context
    }
}
json.dump(output, sys.stdout)
"
