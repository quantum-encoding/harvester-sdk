# Safe Execution Library (safe_exec.so)

## Overview

The `safe_exec.so` library provides **LD_PRELOAD-based command interception** to protect agentic workflows from executing dangerous commands. It intercepts `execvp` calls at the libc level, blocking patterns like `rm -rf /` before they execute.

## How It Works

1. **LD_PRELOAD Hijacking**: The library uses `LD_PRELOAD` to intercept all `execvp()` calls system-wide for any process that loads it.
2. **Pattern Matching**: Before allowing execution, it checks command names and arguments against dangerous patterns.
3. **Transparent Fallback**: Safe commands pass through to the original `execvp()` without modification.

## Protected Patterns

Currently blocks:
- `rm -rf /` - Root filesystem deletion
- `rm -rf ~` - Home directory deletion
- `rm -rf /*` - Wildcard root deletion
- `rm -rf /home/` - Home directory tree deletion
- `rm -rf /etc/` - System config deletion
- `rm -rf /var/` - System data deletion

## Integration

The library is **automatically loaded** by all three agent implementations:

### 1. GrokCodeAgent (`grok_code_agent.py`)
```python
# Enable LD_PRELOAD safety for all subprocess calls
_SAFE_EXEC_LIB = str(Path(__file__).parent / "safe_exec.so")
if os.path.exists(_SAFE_EXEC_LIB):
    os.environ['LD_PRELOAD'] = _SAFE_EXEC_LIB
    logger.info(f"Safe execution library loaded: {_SAFE_EXEC_LIB}")
```

### 2. ClaudeCodeAgent (`claude_code_agent.py`)
Same integration pattern as GrokCodeAgent.

### 3. OpenAIAgent (`openai_agent.py`)
Same integration pattern as GrokCodeAgent.

## Building the Library

Rebuild after modifying `safe_exec.c`:

```bash
cd agents/
gcc -shared -fPIC -o safe_exec.so safe_exec.c -ldl -Wall -Wextra
```

## Testing

Test the protection with a safe directory:

```bash
# Create test directory
mkdir -p /tmp/safe_exec_test
cd /tmp/safe_exec_test

# Load the library
export LD_PRELOAD=/path/to/harvester_sdk/agents/safe_exec.so

# Try dangerous command (should be blocked)
rm -rf /
# Output: Blocked dangerous command: rm -rf /

# Try safe command (should work)
touch testfile.txt
rm testfile.txt
# Output: (file deleted normally)
```

## Extending Protection

Add more dangerous patterns in `safe_exec.c`:

```c
static int is_dangerous(const char *file, char *const argv[]) {
    // Check for dd (disk destroyer)
    if (strstr(file, "dd")) {
        // Check if writing to critical devices
        for (int i = 1; argv[i] != NULL; i++) {
            if (strstr(argv[i], "of=/dev/sda") || strstr(argv[i], "of=/dev/vda")) {
                return 1;  // Dangerous: dd to root disk
            }
        }
    }

    // Existing rm checks...
}
```

After modifications, recompile:
```bash
gcc -shared -fPIC -o safe_exec.so safe_exec.c -ldl
```

## Limitations

- **Coverage**: Only intercepts `execvp`. Other exec variants (`execv`, `execl`, `execle`, etc.) are not covered.
- **Syscall bypass**: Direct `syscall()` usage can bypass this protection.
- **Performance**: Minimal overhead (~µs per exec check).

## Advanced Protection

For production deployments, consider layering additional protections:

1. **Seccomp filters**: Restrict syscalls at kernel level
2. **AppArmor/SELinux**: Mandatory access control policies
3. **Namespace isolation**: Run agents in containerized environments
4. **User permissions**: Run agents as unprivileged users with minimal file access

## Troubleshooting

### Library not loading
```bash
# Check file exists
ls -lh agents/safe_exec.so

# Check for compile errors
gcc -shared -fPIC -o safe_exec.so safe_exec.c -ldl -Wall -Wextra
```

### Commands still executing
```bash
# Verify LD_PRELOAD is set
echo $LD_PRELOAD

# Check if protection is active
python3 -c "import os; print(os.environ.get('LD_PRELOAD'))"
```

### Segmentation faults
```bash
# Debug with gdb
gdb --args python3 your_agent.py
(gdb) run
```

## Security Notice

⚠️ **This is a defense-in-depth layer, not a complete security solution.**

Always follow security best practices:
- Run agents with minimal privileges
- Use read-only filesystems where possible
- Monitor agent execution logs
- Review generated commands before execution in production
- Test in isolated environments first

## License

Copyright (c) 2025 Quantum Encoding Ltd.
Part of the Harvester SDK security infrastructure.
