# Harvester SDK Security Architecture

## 🛡️ The Warden's Arsenal - Multi-Layer Injection Protection

The Harvester SDK implements **defense-in-depth** security for agentic workflows, protecting against prompt injection attacks that could execute malicious commands. This security infrastructure operates at multiple levels:

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Prompt Sanitization (Unicode/Emoji Smuggling)    │
│  → PromptSanitizer: NFKD normalization, emoji stripping     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Agent-Level Validation                            │
│  → Input validation, output schema checking                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: LD_PRELOAD Syscall Interception                   │
│  → safe_exec.so: execvp() hook (10 threat categories)       │
│  → safe_open.so: open()/fopen() hook (file write blocking)  │
│  → safe_fork.so: fork()/vfork() hook (rate limiting)        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔒 Layer 1: Prompt Sanitization

**Module:** `agents/prompt_sanitizer.py`

Defends against Unicode smuggling, emoji injection, and link smuggling attacks.

### Features

1. **Advanced Unicode Fixing (ftfy)**
   - Fixes mojibake, broken encodings, and Unicode corruption
   - Example: `Ã©` → `é`, broken surrogate pairs fixed
   - Requires: `pip install ftfy`

2. **Unicode Normalization (NFKD)**
   - Converts fancy Unicode variants to ASCII equivalents
   - Example: `ｒｍ` (fullwidth) → `rm`, `𝗿𝗺` (bold) → `rm`

3. **Zero-Width Character Removal**
   - Strips: `\u200B` (ZWSP), `\u200C` (ZWNJ), `\uFEFF` (BOM), etc.
   - Prevents: `rm\u200B-rf\u200C/` attacks

4. **Emoji Stripping**
   - Removes all emoji sequences
   - Prevents emoji-based command smuggling

5. **URL Validation**
   - Blocks dangerous schemes: `javascript:`, `data:`, `blob:`, `file:`
   - Optional HTTP/HTTPS blocking

6. **Pattern Detection**
   - Base64-like strings (potential encoded commands)
   - Suspicious command patterns (`rm -rf /`, `curl | bash`, etc.)

### Usage

```python
from agents import PromptSanitizer, sanitize_prompt

# Quick sanitization
clean_prompt = sanitize_prompt(user_input)

# Full analysis
result = PromptSanitizer.sanitize(
    prompt=user_input,
    aggressive=True,  # Strip all non-ASCII
    allow_http=False  # Block HTTP URLs
)

if not result['safe']:
    print(f"Blocked URLs: {result['blocked_urls']}")
    print(f"Detected patterns: {result['detected_patterns']}")
```

---

## 🚫 Layer 2: Syscall Interception (LD_PRELOAD)

### safe_exec.so - Command Execution Protection

**Source:** `agents/safe_exec.c` (350+ lines)
**Blocks:** 11 threat categories

#### Protected Threat Categories

1. **Disk/Filesystem Destruction**
   - `rm -rf /`, `dd of=/dev/sda`, `mkfs`, `fdisk`, `shred -r`
   - `mv * /dev/null`

2. **Privilege Escalation**
   - `sudo`, `su`, `pkexec`
   - `chmod -R /`, `chown -R /`
   - `useradd`, `passwd`, `groupmod`

3. **Resource Exhaustion**
   - Fork bombs: `:(){:|:&};:`
   - Infinite loops: `while true`
   - Disk fillers: `yes > /dev/sda`

4. **Network/External Threats**
   - `curl | bash`, `wget | sh`
   - `ssh root@`, `scp root@`

5. **System Disruption**
   - `reboot`, `shutdown`, `halt`
   - `systemctl stop/disable`
   - `iptables -F`, `sysctl kernel.*`

6. **Obfuscation Techniques**
   - `base64 -d | bash`
   - `echo | bash`, `eval`, `exec`

7. **Backdoors & Reverse Shells**
   - `nc -lvp`, `netcat -e /bin/bash`
   - `bash -i >& /dev/tcp/`
   - Crypto miners: `xmrig`, `minerd`, `cpuminer`

8. **Environment Variable Manipulation**
   - `export PATH=/tmp:$PATH`
   - `unset HISTFILE`, `HISTFILESIZE=0`
   - Unauthorized `LD_PRELOAD=`

9. **Kernel & Module Exploits**
   - `insmod`, `modprobe`, `rmmod`
   - `/dev/kmem`, `/dev/mem` access
   - Kernel compilation

10. **Scripted Injections** (NEW)
    - Python: `python -c 'import os; os.system("rm -rf /")'`
    - Perl: `perl -e 'system("rm -rf /")'`
    - Ruby: `ruby -e 'system("rm -rf /")'`
    - Node.js: `node -e 'require("child_process").exec("rm -rf /")'`

11. **Package Management** (Optional)
    - `apt remove`, `yum erase`, etc.

#### Environment Variables

```bash
# Enable logging
export SAFE_EXEC_LOG=1

# Emergency bypass (NOT RECOMMENDED)
export SAFE_EXEC_OVERRIDE=1
```

---

### safe_open.so - File Write Protection

**Source:** `agents/safe_open.c` (170+ lines)
**Intercepts:** `open()`, `fopen()`

#### Protected Paths

- **Block Devices:** `/dev/sda`, `/dev/vda`, `/dev/nvme0n1`
- **System Config:** `/etc/`, `/boot/`, `/sys/`, `/proc/`
- **Kernel Parameters:** `/proc/sys/kernel/`, `/proc/sys/vm/`, `/proc/sys/net/`, `/proc/sysrq-trigger` (NEW)
- **Auth Files:** `/etc/passwd`, `/etc/shadow`, `/etc/sudoers`
- **Critical Configs:** `/etc/fstab`, `/etc/hosts`, `/etc/resolv.conf`
- **Init Systems:** `/etc/systemd/`, `/lib/systemd/`

#### Environment Variables

```bash
# Enable logging
export SAFE_OPEN_LOG=1

# Emergency bypass
export SAFE_OPEN_OVERRIDE=1
```

---

### safe_fork.so - Fork Bomb Protection

**Source:** `agents/safe_fork.c` (180+ lines)
**Intercepts:** `fork()`, `vfork()`

#### Features

- **Rate Limiting:** Max 10 forks/second (default)
- **Total Cap:** Max 1000 forks/process (default)
- **Idle Reset Timer:** Rate counter resets after 5 seconds of inactivity (NEW)
- **Auto-Detection:** Catches `:(){:|:&};:` and similar bombs

#### Environment Variables

```bash
# Configure limits
export SAFE_FORK_MAX_PER_SEC=10
export SAFE_FORK_MAX_TOTAL=1000
export SAFE_FORK_IDLE_RESET=5  # Seconds of idle before rate reset (NEW)

# Enable logging
export SAFE_FORK_LOG=1

# Emergency bypass
export SAFE_FORK_OVERRIDE=1
```

---

## 🔧 Integration

### Automatic Loading

All three agent implementations (`GrokCodeAgent`, `ClaudeCodeAgent`, `OpenAIAgent`) automatically load the safety libraries on import:

```python
# In agents/grok_code_agent.py (and others)
import os
from pathlib import Path

_SAFE_LIBS = [
    Path(__file__).parent / "safe_exec.so",
    Path(__file__).parent / "safe_open.so",
    Path(__file__).parent / "safe_fork.so",
]

libs = [str(lib) for lib in _SAFE_LIBS if lib.exists()]
if libs:
    os.environ['LD_PRELOAD'] = ':'.join(libs)
```

### Manual Loading

```python
import os
os.environ['LD_PRELOAD'] = '/path/to/safe_exec.so:/path/to/safe_open.so:/path/to/safe_fork.so'
```

---

## 🧪 Testing

### Comprehensive Test Suite

**Script 1:** `agents/test_injection_protection.py`

Runs 30+ injection attack simulations across 8 categories:

```bash
python3 agents/test_injection_protection.py
```

**Categories Tested:**
- Disk/Filesystem Destruction (5 tests)
- Privilege Escalation (3 tests)
- Resource Exhaustion (2 tests)
- Network Threats (4 tests)
- Obfuscation Techniques (2 tests)
- Backdoors & Miners (2 tests)
- Environment Manipulation (2 tests)
- Safe Commands Control (3 tests)

**Script 2:** `agents/test_injection_fuzzing.py` (NEW)

Hypothesis-based property testing with randomized obfuscation:

```bash
pip install hypothesis
python3 agents/test_injection_fuzzing.py
```

**Fuzzing Tests:**
- Unicode obfuscation (50 examples per test)
- Zero-width character injection (50 examples)
- Emoji smuggling (50 examples)
- Base64 detection (20 examples)
- Multi-layer obfuscation (30 examples)
- Scripted injection blocking (20 examples)

### Example Test Output

```
================================================================================
Disk/Filesystem Destruction
================================================================================

Testing: rm -rf root
  Attempt to delete root filesystem
  Command: rm -rf /...
  ✓ PASS - Blocked as expected
  Error: [SAFE_EXEC] ⛔ BLOCKED DANGEROUS COMMAND

Testing: Safe rm
  Safe file deletion in /tmp
  Command: cd /tmp && touch test.txt && rm test.txt...
  ✓ PASS - Allowed as expected
```

### Security Log Monitoring (NEW)

**Script:** `agents/security_monitor.py`

Analyze logs from safe_exec/open/fork to detect attack patterns:

```bash
# Analyze log file
python3 agents/security_monitor.py /var/log/agent.log

# Read from stdin
tail -f /var/log/agent.log | python3 agents/security_monitor.py

# Export as JSON
python3 agents/security_monitor.py /var/log/agent.log --json
```

**Report Features:**
- Attack pattern detection (7 categories)
- Top blocked commands/files
- Fork violation analysis
- Security score and threat level assessment

---

## 🏗️ Building from Source

### Prerequisites

```bash
sudo apt install build-essential gcc
```

### Compile Libraries

```bash
cd agents/

# Compile all three libraries
gcc -shared -fPIC -o safe_exec.so safe_exec.c -ldl -Wall -Wextra
gcc -shared -fPIC -o safe_open.so safe_open.c -ldl -Wall -Wextra
gcc -shared -fPIC -o safe_fork.so safe_fork.c -ldl -Wall -Wextra

# Verify
ls -lh safe_*.so
```

---

## 📦 Distribution

The safety libraries are **automatically included** in the PyPI distribution:

```bash
pip install harvester-sdk
```

The `.so` files and `.c` sources are bundled via `MANIFEST.in`:

```
include agents/safe_exec.so
include agents/safe_open.so
include agents/safe_fork.so
include agents/safe_exec.c
include agents/safe_open.c
include agents/safe_fork.c
```

Users can rebuild for their architecture if needed.

---

## ⚠️ Limitations & Caveats

### Coverage

- **Exec Variants:** Only `execvp` is intercepted. Direct syscalls (`syscall(SYS_execve, ...)`) bypass this.
- **Other Languages:** Protection is specific to C-based `libc` calls. Pure Python subprocesses are covered, but native Go/Rust binaries may not be.

### Performance

- **Overhead:** Negligible (~microseconds per call for pattern matching)
- **Tested:** No measurable impact on typical agent workloads

### False Positives

- Some legitimate operations may be blocked (e.g., `systemctl stop my-service`)
- Use `SAFE_EXEC_OVERRIDE=1` sparingly for troubleshooting

---

## 🚀 Production Deployment

### Recommended Configuration

```bash
# In production, enable logging for auditing
export SAFE_EXEC_LOG=1
export SAFE_OPEN_LOG=1
export SAFE_FORK_LOG=1

# Disable overrides (force protection)
unset SAFE_EXEC_OVERRIDE
unset SAFE_OPEN_OVERRIDE
unset SAFE_FORK_OVERRIDE
```

### Layered Security

For production, combine with:

1. **Seccomp Filters:** Kernel-level syscall restrictions
2. **AppArmor/SELinux:** Mandatory access control
3. **Container Isolation:** Run agents in Docker/Kubernetes with limited privileges
4. **User Permissions:** Run as unprivileged user with minimal file access

---

## 📚 Research & References

This implementation addresses real-world attack vectors documented in:

- **OWASP Top 10 for LLM Applications**
- **Prompt Injection Research:** Unicode smuggling, emoji injection, link smuggling
- **CTF Challenge Patterns:** Fork bombs, reverse shells, crypto miners
- **Red Team Playbooks:** Common evasion techniques

---

## 🔐 Security Disclosure

If you discover a bypass or vulnerability in the protection mechanisms, please report responsibly:

- **Email:** security@quantumencoding.io
- **PGP:** Available on request

Do not publicly disclose vulnerabilities until patched.

---

## 📄 License

Copyright (c) 2025 Quantum Encoding Ltd.
Part of the Harvester SDK security infrastructure.

This security architecture is provided "as-is" without warranty. While it significantly reduces risk, no system is 100% secure. Always follow security best practices and defense-in-depth principles.

---

## 🎯 Quick Reference

| Library | Protects Against | LOC | Env Vars |
|---------|------------------|-----|----------|
| `safe_exec.so` | Command injection (11 categories) | 350+ | `SAFE_EXEC_LOG`, `SAFE_EXEC_OVERRIDE` |
| `safe_open.so` | File write attacks (/proc/ blocking) | 170+ | `SAFE_OPEN_LOG`, `SAFE_OPEN_OVERRIDE` |
| `safe_fork.so` | Fork bombs (with idle reset) | 180+ | `SAFE_FORK_LOG`, `SAFE_FORK_OVERRIDE`, `SAFE_FORK_MAX_PER_SEC`, `SAFE_FORK_MAX_TOTAL`, `SAFE_FORK_IDLE_RESET` |
| `prompt_sanitizer.py` | Unicode/emoji smuggling (ftfy) | 300+ | N/A (Python module) |
| `security_monitor.py` | Log analysis & attack detection | 270+ | N/A (CLI tool) |
| `test_injection_fuzzing.py` | Hypothesis-based fuzzing | 250+ | N/A (Test suite) |

**Total Protection:** 1500+ lines of security code, 250+ test examples, 11 threat categories

---

*For more details, see:*
- `agents/README_SAFE_EXEC.md` - Original implementation guide
- `agents/test_injection_protection.py` - Comprehensive test suite
- `agents/test_injection_fuzzing.py` - Hypothesis fuzzing tests
- `agents/security_monitor.py` - Log analysis tool
- `agents/prompt_sanitizer.py` - Prompt sanitization module

---

## 🔮 Roadmap

**Planned Enhancements:**
- WebAssembly (WASM) sandboxing for untrusted code execution
- Real-time security dashboard with Prometheus/Grafana integration
- Machine learning-based anomaly detection for novel attack patterns
- Extended language support (Go, Rust subprocess interception)
- Cloud deployment templates (AWS Lambda layers, Docker base images)
