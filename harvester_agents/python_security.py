#!/usr/bin/env python3
"""
PROJECT PURGE: Python Security Interceptor
Prevents malicious Python code execution in agent shells.
© 2025 Quantum Encoding Ltd - Security Division
"""

import re
import ast
import subprocess
from typing import Tuple, Optional

# Forbidden modules that can be used for file system destruction
FORBIDDEN_MODULES = {
    'shutil',     # shutil.rmtree() bypasses safe_exec
    'pathlib',    # Path.unlink() bypasses safe_exec
    'glob',       # Can be used to enumerate targets for deletion
}

# Forbidden function calls
FORBIDDEN_CALLS = {
    'os.system',      # Direct shell execution
    'os.remove',      # File deletion
    'os.unlink',      # File deletion
    'os.rmdir',       # Directory deletion
    'os.removedirs',  # Recursive directory deletion
    'subprocess.call', # Shell execution
    'eval',           # Code execution
    'exec',           # Code execution
    '__import__',     # Dynamic imports
}

# Forbidden patterns in shell commands
FORBIDDEN_SHELL_PATTERNS = [
    r'rm\s+-.*rf',              # rm -rf or rm -fr
    r'rm\s+.*\/home\/rich',     # Any rm targeting home
    r'dd\s+.*of=/dev/',         # dd to devices
    r'shutil\.rmtree',          # Python shutil.rmtree
    r'os\.remove',              # Python os.remove
    r'os\.unlink',              # Python os.unlink
    r'Path\(.*\)\.unlink',      # pathlib unlink
    r'\/home\/rich\/(Downloads|Music|Documents|github)', # Critical directories
]


class SecurityViolation(Exception):
    """Raised when malicious code is detected."""
    pass


def analyze_python_code(code: str) -> Tuple[bool, Optional[str]]:
    """
    Analyze Python code for malicious patterns.

    Returns:
        (is_safe, violation_message)
    """
    # Check for forbidden modules in imports
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in FORBIDDEN_MODULES:
                        return False, f"Forbidden module import: {alias.name}"

            elif isinstance(node, ast.ImportFrom):
                if node.module in FORBIDDEN_MODULES:
                    return False, f"Forbidden module import: {node.module}"

            # Check function calls
            elif isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        func_name = f"{node.func.value.id}.{node.func.attr}"
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id

                if func_name in FORBIDDEN_CALLS:
                    return False, f"Forbidden function call: {func_name}"

    except SyntaxError:
        # If it's not valid Python, treat it as shell command
        pass

    # Check for forbidden patterns (regex-based)
    for pattern in FORBIDDEN_SHELL_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            return False, f"Forbidden pattern detected: {pattern}"

    return True, None


def secure_shell_execute(command: str, timeout: int = 30) -> str:
    """
    Execute shell command with security checks.

    This is THE PYTHON INTERCEPTOR - all agent shell executions must go through this.

    Args:
        command: Shell command to execute
        timeout: Execution timeout in seconds

    Returns:
        Command output or error message

    Raises:
        SecurityViolation: If malicious code is detected
    """
    # Security Analysis
    is_safe, violation = analyze_python_code(command)

    if not is_safe:
        error_msg = f"""
🚨 SECURITY VIOLATION DETECTED 🚨
Command: {command[:100]}...
Reason: {violation}

This command has been BLOCKED by the Python Security Interceptor.
If this is a false positive, please review the security policy.
"""
        print(error_msg)
        raise SecurityViolation(violation)

    # If safe, execute with subprocess
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        output = f"Command: {command}\n"
        output += f"Exit code: {result.returncode}\n"

        if result.stdout:
            output += f"\nOutput:\n{result.stdout}"
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"

        return output

    except subprocess.TimeoutExpired:
        return f"✗ Command timed out after {timeout} seconds: {command}"
    except Exception as e:
        return f"✗ Error executing command: {str(e)}"


def test_security():
    """Test the security interceptor."""
    print("🛡️ Testing Python Security Interceptor...")
    print()

    # Safe commands
    safe_tests = [
        "ls -la",
        "echo 'Hello World'",
        "python -c \"print('test')\"",
        "gcc -o test test.c",
    ]

    # Malicious commands
    malicious_tests = [
        "rm -rf /home/rich/Downloads",
        "shutil.rmtree('/home/rich/Music')",
        "dd if=/dev/zero of=/dev/sda",
        "python -c \"import shutil; shutil.rmtree('/home/rich')\"",
        "os.system('rm -rf /')",
    ]

    print("Testing SAFE commands:")
    for cmd in safe_tests:
        is_safe, msg = analyze_python_code(cmd)
        status = "✅ PASS" if is_safe else "❌ FAIL"
        print(f"  {status}: {cmd}")

    print("\nTesting MALICIOUS commands (should be blocked):")
    for cmd in malicious_tests:
        is_safe, msg = analyze_python_code(cmd)
        status = "✅ BLOCKED" if not is_safe else "❌ ALLOWED (BUG!)"
        print(f"  {status}: {cmd}")
        if msg:
            print(f"       Reason: {msg}")


if __name__ == "__main__":
    test_security()
