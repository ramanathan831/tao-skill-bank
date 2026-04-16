#!/usr/bin/env bash
# PostCompact hook: re-inject DEFT loop state + iteration summaries into context.
# Finds the most recently modified deft_state.json under the workspace,
# reads it + all *_summary.md files + the latest RCA report path.
# Outputs JSON with additionalContext for Claude to consume.

set -euo pipefail

WORKSPACE="${DEFT_WORKSPACE:-$PWD}"

# Find the most recently modified deft_state.json
STATE_FILE=$(find "$WORKSPACE" -name "deft_state.json" -not -path "*/node_modules/*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$STATE_FILE" ]; then
  # No active DEFT loop — nothing to restore
  exit 0
fi

RESULTS_DIR=$(dirname "$STATE_FILE")

# Build the context string
CONTEXT="[PostCompact: DEFT loop state restored automatically]\n\n"
CONTEXT+="## Current State (deft_state.json)\n\`\`\`json\n"
CONTEXT+="$(cat "$STATE_FILE")"
CONTEXT+="\n\`\`\`\n\n"

# Append all iteration summaries in order
SUMMARIES=$(ls "$RESULTS_DIR"/*_summary.md 2>/dev/null || true)
if [ -n "$SUMMARIES" ]; then
  CONTEXT+="## Iteration Summaries\n\n"
  for f in $SUMMARIES; do
    CONTEXT+="$(cat "$f")\n\n---\n\n"
  done
fi

# Find the most recent RCA report path from state file
LATEST_RCA=$(python3 -c "
import json, sys
with open('$STATE_FILE') as f:
    state = json.load(f)
iters = state.get('iterations', {})
for key in reversed(sorted(iters.keys())):
    rca = iters[key].get('rca_report', '')
    if rca:
        print(rca)
        break
" 2>/dev/null || true)

if [ -n "$LATEST_RCA" ] && [ -f "$LATEST_RCA" ]; then
  CONTEXT+="## Latest RCA Report\nPath: $LATEST_RCA\n"
  CONTEXT+="(Re-read this file with the Read tool for full details before making strategy decisions.)\n"
fi

# Output JSON for Claude Code hook system
python3 -c "
import json, sys
context = '''$(echo -e "$CONTEXT")'''
output = {
    'hookSpecificOutput': {
        'hookEventName': 'PostCompact',
        'additionalContext': context
    }
}
json.dump(output, sys.stdout)
"
