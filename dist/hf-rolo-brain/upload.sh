#!/usr/bin/env bash
# Stage 11 publish runner — uploads rolo-brain artifacts to HuggingFace.
#
# Per plans/rolo-finetune-ship-v2.md §11.4 the NOTICE acceptance gate
# requires you to have read NOTICE before this script runs. The first prompt
# confirms that.
#
# Usage:  ./dist/hf-rolo-brain/upload.sh
# Re-run safe: `hf upload` overwrites in place; the script is idempotent.

set -euo pipefail

# Pipx installs `hf` to ~/.local/bin which is on the interactive zsh PATH
# (via .zshrc) but not always on the non-interactive bash PATH this script
# runs under. Override via `HF_BIN=/some/path/hf ./upload.sh` if needed.
HF_BIN="${HF_BIN:-$HOME/.local/bin/hf}"
if [ ! -x "$HF_BIN" ]; then
  HF_BIN="$(command -v hf 2>/dev/null || true)"
fi
if [ -z "$HF_BIN" ] || [ ! -x "$HF_BIN" ]; then
  echo "FAIL: cannot find the 'hf' binary. Install with 'pipx install huggingface_hub'."
  exit 1
fi

REPO="larawashington/rolo-brain"
EXPECTED_USER="${REPO%%/*}"
STAGING="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$STAGING/../.." && pwd)"
GGUF_SRC="$ROOT/build/rolo-v2.1-Q5_K_M.gguf"
GGUF_DEST="rolo-brain-Q5_K_M.gguf"

# Single source of truth for the small-file artifact set. Local filename
# in $STAGING == remote filename at the HF repo root. Parallel array
# ARTIFACT_MSGS holds the per-file commit message.
ARTIFACTS=(NOTICE README.md Modelfile template params runtime_contract.md)
ARTIFACT_MSGS=(
  "add NOTICE"
  "add model card"
  "add Ollama Modelfile (local-create reference)"
  "add Ollama HF-pull chat template"
  "add Ollama HF-pull sampling params"
  "add runtime contract"
)

echo "→ Pre-flight checks"
for f in "${ARTIFACTS[@]}"; do
  test -f "$STAGING/$f" || { echo "FAIL: $f not found in $STAGING"; exit 1; }
done
test -f "$GGUF_SRC" || { echo "FAIL: GGUF not found at $GGUF_SRC"; exit 1; }

WHOAMI_RAW="$("$HF_BIN" auth whoami 2>&1 || true)"
HF_USER="$(printf '%s\n' "$WHOAMI_RAW" | sed -n 's/^[[:space:]]*user[:=][[:space:]]*//p' | head -1)"
if [ -z "$HF_USER" ]; then
  echo "FAIL: could not parse a user from 'hf auth whoami'."
  echo "      Raw output (stdout+stderr):"
  printf '%s\n' "$WHOAMI_RAW" | sed 's/^/        /'
  echo "      If you are not logged in, run:  $HF_BIN auth login"
  exit 1
fi
[ "$HF_USER" = "$EXPECTED_USER" ] || { echo "FAIL: expected HF user '$EXPECTED_USER', got '$HF_USER'"; exit 1; }

GGUF_SHA="$(shasum -a 256 "$GGUF_SRC" | awk '{print $1}')"
MODELFILE_SHA="$(shasum -a 256 "$STAGING/Modelfile" | awk '{print $1}')"
TEMPLATE_SHA="$(shasum -a 256 "$STAGING/template"  | awk '{print $1}')"
PARAMS_SHA="$(shasum -a 256 "$STAGING/params"      | awk '{print $1}')"
echo "    GGUF sha256:  $GGUF_SHA"
echo "    GGUF size:    $(du -h "$GGUF_SRC" | awk '{print $1}')"

echo
echo "→ Acceptance gate (plans/rolo-finetune-ship-v2.md §11.4)"
echo "  Have you read dist/hf-rolo-brain/NOTICE? It must reflect the actual"
echo "  training-data provenance for v2.1 and cite the Gemma Terms of Use."
read -p "  Type 'yes' to confirm: " confirm
[ "$confirm" = "yes" ] || { echo "Aborted — re-read NOTICE then re-run."; exit 1; }

echo
echo "→ Create repo (idempotent — --exist-ok no-ops on an existing repo)"
"$HF_BIN" repos create "$REPO" --type model --exist-ok >/dev/null 2>&1 || true

echo
echo "→ Upload artifacts to $REPO"
for i in "${!ARTIFACTS[@]}"; do
  f="${ARTIFACTS[$i]}"
  msg="${ARTIFACT_MSGS[$i]}"
  "$HF_BIN" upload "$REPO" "$STAGING/$f" "$f" --commit-message "$msg"
done
"$HF_BIN" upload "$REPO" "$GGUF_SRC" "$GGUF_DEST" --commit-message "add Q5_K_M GGUF (Stage 9)"

echo
echo "→ Post-upload verification"
echo "  1. From a clean Ollama state (or a second machine):"
echo "       ollama pull hf.co/$REPO"
echo "  2. Modelfile parity diff:"
echo "       ollama show hf.co/$REPO --modelfile | diff - $STAGING/Modelfile"
echo "     (FROM-line will differ — that is expected; template + stops must match.)"
echo "  3. Re-run Stage 10 verification #2 (golden fixture) and #3 (<idle> sentinel)"
echo "     against the HF-pulled model. See plans/rolo-finetune-ship-v2.md §11.4."

echo
echo "→ Recording publish manifest"
MANIFEST="$ROOT/data/finetune/sft-v2/manifests/11_publish.json"
UPLOADS_JSON="$(
  for f in "${ARTIFACTS[@]}" "$GGUF_DEST"; do
    printf '    "%s",\n' "$f"
  done | sed '$s/,$//'
)"
cat > "$MANIFEST" <<EOF
{
  "step": "11_publish",
  "finished_at": "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)",
  "hf_repo": "$REPO",
  "hf_url": "https://huggingface.co/$REPO",
  "gguf": {
    "local_path": "$GGUF_SRC",
    "hf_filename": "$GGUF_DEST",
    "sha256": "$GGUF_SHA"
  },
  "modelfile_sha256": "$MODELFILE_SHA",
  "template_sha256": "$TEMPLATE_SHA",
  "params_sha256": "$PARAMS_SHA",
  "uploads": [
$UPLOADS_JSON
  ],
  "notice_acceptance_confirmed_by_lara": true,
  "modelfile_parity_post_pull": null,
  "idle_sentinel_reverify_post_pull": null
}
EOF
echo "    wrote $MANIFEST"
echo "    (set modelfile_parity_post_pull + idle_sentinel_reverify_post_pull after step 2/3 above)"

echo
echo "✓ Stage 11 upload finished. Next: do the post-upload verification above."
