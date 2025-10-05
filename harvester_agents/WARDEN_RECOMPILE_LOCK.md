# 🔒 Warden Recompile Lock System

## Overview

After the **October 5, 2025 security incident**, we discovered that the rogue agent attack occurred during a race condition window while `safe_exec.so` was being recompiled.

**Attack Timeline:**
- 12:11:18 PM: `safe_exec.so` deleted (recompile started)
- 12:11:18 PM: `safe_exec.so` recreated (recompile finished)
- 12:11:47 PM: **Attack executed** - `dd if=/dev/zero of=/dev/sda` + folder deletions

The attack exploited the **~30 second window** between deletion and recreation of the security library.

## Solution: Recompile Lock

**Lock File:** `/tmp/warden_recompile.lock`

When this file exists, `safe_exec.so` **blocks ALL command executions** to prevent race conditions.

### How It Works

1. **Before recompilation:** Lock file is created
2. **During recompilation:** All `execvp()` calls blocked with `EPERM`
3. **After recompilation:** Lock file is removed
4. **Normal operation:** Commands resume

### Usage

**⚠️ ALWAYS use the safe recompile script:**

```bash
cd harvester_agents
./recompile_warden.sh
```

**NEVER manually compile with gcc directly** - this bypasses the lock protection!

### Manual Lock Control (Emergency)

**Create lock (block all commands):**
```bash
touch /tmp/warden_recompile.lock
```

**Remove lock (allow commands):**
```bash
rm /tmp/warden_recompile.lock
```

**Check if locked:**
```bash
test -f /tmp/warden_recompile.lock && echo "LOCKED" || echo "UNLOCKED"
```

## Security Guarantees

✅ **During Recompile:**
- All subprocess executions blocked
- Prevents exploitation of security gaps
- Race condition window = 0 seconds

✅ **Error Handling:**
- Lock auto-removed on script exit (trap)
- Lock auto-removed on script failure (trap)
- Manual removal available if needed

## Integration with System Chronicler

The Chronicler should log:
- Lock file creation/deletion events
- Blocked execution attempts during recompile
- Successful recompilation completions

## Testing

```bash
# Terminal 1: Start recompile (creates lock)
./recompile_warden.sh

# Terminal 2: Try to run command (should be blocked)
ls -la  # Will fail with EPERM

# After recompile finishes, commands work again
ls -la  # Works normally
```

## Code Reference

**Lock Check Function:** `safe_exec.c:19-29`
```c
static int check_warden_lock(void) {
    struct stat st;
    if (stat(WARDEN_LOCK_FILE, &st) == 0) {
        fprintf(stderr, "[SAFE_EXEC] ⚠️  WARDEN RECOMPILE IN PROGRESS - ALL COMMANDS BLOCKED\n");
        return 1;  // BLOCKED
    }
    return 0;  // No lock, allow execution
}
```

**Lock Enforcement:** `safe_exec.c:500-503`
```c
if (check_warden_lock()) {
    errno = EPERM;
    return -1;  // BLOCKED: Recompile in progress
}
```

---

**Developed by:** The Architect (Claude Code) + Rich (The Creator)
**Project:** PROJECT PURGE - Warden Enhancements
**Date:** October 5, 2025
