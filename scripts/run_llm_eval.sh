#!/usr/bin/env bash
# Real-LLM evaluation driver (§7.2 multi-model): loads each downloaded model in
# LM Studio in turn and runs `pdam llm-eval` across all workloads, all attacks,
# {none, minimal_defense}, with repetitions. Results land in results/llm/<model>/.
#
# Usage:  bash scripts/run_llm_eval.sh
# Env:    MODELS, REPEATS, TEMP, DEFENSES, DIFFICULTY can override the defaults.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
LMS="${LMS:-$HOME/.lmstudio/bin/lms}"

# open-weight models spanning families/sizes (Meta / Mistral / Google)
DEFAULT_MODELS="meta-llama-3.1-8b-instruct mistral-7b-instruct-v0.3 mistral-small-24b-instruct-2501 gemma-2-27b-it"
read -r -a MODELS <<< "${MODELS:-$DEFAULT_MODELS}"
REPEATS="${REPEATS:-3}"
TEMP="${TEMP:-0.4}"
DEFENSES="${DEFENSES:-none,minimal_defense}"
DIFFICULTY="${DIFFICULTY:-easy}"

echo "models=${MODELS[*]}  repeats=$REPEATS  temp=$TEMP  defenses=$DEFENSES  difficulty=$DIFFICULTY"
"$LMS" server start >/dev/null 2>&1 || true

for m in "${MODELS[@]}"; do
  safe="${m//\//_}"
  echo; echo "############### $m ###############"
  "$LMS" unload --all >/dev/null 2>&1 || true
  if ! timeout 240 "$LMS" load "$m" --context-length 8192 -y >/dev/null 2>&1; then
    echo "LOAD FAILED: $m — skipping"; continue
  fi
  python3 -m pdam llm-eval --model "$m" --workloads all \
      --defenses "$DEFENSES" --difficulty "$DIFFICULTY" \
      --repeats "$REPEATS" --temperature "$TEMP" \
      --outdir "results/llm/$safe"
done

"$LMS" unload --all >/dev/null 2>&1 || true
python3 scripts/combine_llm_results.py results/llm
echo; echo "DONE. See results/llm/RESULTS.md"
