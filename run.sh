#!/usr/bin/env bash
set -euo pipefail
command="${1:-}"
case "$command" in
  prepare|train|predict|evaluate|rank|reproduce) shift ;;
  *) printf '%s\n' 'Usage: bash run.sh {prepare|train|predict|evaluate|rank|reproduce} ...' >&2; exit 1 ;;
esac
exec python3 "$(dirname "$0")/$command.py" "$@"
