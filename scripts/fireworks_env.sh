#!/usr/bin/env bash
# Export FIREWORKS_API_KEY from SSM Parameter Store.
#
# The key lives in SSM, not in a file and not in a shell history, because an
# access key already leaked into a chat transcript once on this project. SSM is
# the same place `/vektori/modal-toml` already lives, so there is one secret
# store to audit and rotate rather than two.
#
# Put it there first (run this yourself — the value must not pass through an
# agent transcript):
#
#   aws ssm put-parameter --name /vektori/fireworks-key --type SecureString \
#     --value "fw_..." --overwrite
#
# Then, on the box or locally:
#
#   source scripts/fireworks_env.sh
#   uv run vektori-trace <teacher command>
#
# Sourced, not executed: an exported variable cannot survive a subshell.

PARAM_NAME="${VEKTORI_FIREWORKS_PARAM:-/vektori/fireworks-key}"

if ! command -v aws >/dev/null 2>&1; then
  echo "fireworks_env: aws CLI not on PATH (try PATH=\$HOME/.local/bin:\$PATH)" >&2
  return 1 2>/dev/null || exit 1
fi

# --with-decryption is what makes a SecureString readable; without it the call
# succeeds and hands back ciphertext, which fails later as a 401 that looks like
# a bad key rather than a bad fetch.
_fw_key="$(aws ssm get-parameter --name "$PARAM_NAME" --with-decryption \
             --query 'Parameter.Value' --output text 2>/dev/null)"

if [ -z "$_fw_key" ] || [ "$_fw_key" = "None" ]; then
  echo "fireworks_env: could not read $PARAM_NAME from SSM." >&2
  echo "  put it there with:" >&2
  echo "  aws ssm put-parameter --name $PARAM_NAME --type SecureString --value 'fw_...' --overwrite" >&2
  unset _fw_key
  return 1 2>/dev/null || exit 1
fi

export FIREWORKS_API_KEY="$_fw_key"
unset _fw_key

# Length only. Never the value, and not even a prefix — this output lands in
# logs and transcripts.
echo "fireworks_env: FIREWORKS_API_KEY set from $PARAM_NAME (${#FIREWORKS_API_KEY} chars)"
