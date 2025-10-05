#!/bin/bash
# Safe Warden Recompile Script
# Creates lock file to prevent race conditions during recompilation

set -e

LOCK_FILE="/tmp/warden_recompile.lock"
AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔒 WARDEN RECOMPILE: Creating lock file..."
touch "$LOCK_FILE"

# Ensure lock is removed on exit (even if script fails)
trap "rm -f $LOCK_FILE; echo '🔓 WARDEN RECOMPILE: Lock released'" EXIT

echo "🔨 Compiling safe_exec.c..."
gcc -shared -fPIC -o "$AGENT_DIR/safe_exec.so" "$AGENT_DIR/safe_exec.c" -ldl

echo "🔨 Compiling safe_open.c..."
gcc -shared -fPIC -o "$AGENT_DIR/safe_open.so" "$AGENT_DIR/safe_open.c" -ldl

echo "🔨 Compiling safe_fork.c..."
gcc -shared -fPIC -o "$AGENT_DIR/safe_fork.so" "$AGENT_DIR/safe_fork.c" -ldl

echo "✅ WARDEN RECOMPILE: Complete!"
echo "🛡️  All security libraries updated successfully"
