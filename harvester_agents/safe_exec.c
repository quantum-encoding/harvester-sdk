#include <dlfcn.h>    // For dlsym
#include <errno.h>    // For error codes
#include <stdarg.h>   // For va_list
#include <stdio.h>    // For fprintf
#include <string.h>   // For strstr, strcmp
#include <unistd.h>   // For execvp prototype
#include <stdlib.h>   // For getenv

// Original execvp function pointer
static int (*original_execvp)(const char *file, char *const argv[]) = NULL;

// ========================================================================
// PROJECT WARDEN: THE SCRIBE'S PASS
// Context-aware security policy for legitimate compilation
// ========================================================================

// Helper to get parent process name
static int get_parent_process_name(char *name, size_t size) {
    FILE *fp;
    char path[256];
    char cmdline[1024];
    pid_t ppid;

    // Read our parent PID from /proc/self/stat
    fp = fopen("/proc/self/stat", "r");
    if (!fp) return -1;

    // Format: pid (comm) state ppid ...
    // We need the 4th field (ppid)
    if (fscanf(fp, "%*d %*s %*c %d", &ppid) != 1) {
        fclose(fp);
        return -1;
    }
    fclose(fp);

    // Read parent's command line
    snprintf(path, sizeof(path), "/proc/%d/cmdline", ppid);
    fp = fopen(path, "r");
    if (!fp) return -1;

    size_t len = fread(cmdline, 1, sizeof(cmdline) - 1, fp);
    fclose(fp);

    if (len == 0) return -1;
    cmdline[len] = '\0';

    // Extract just the executable name (last component of path)
    char *basename = strrchr(cmdline, '/');
    if (basename) {
        basename++; // Skip the '/'
    } else {
        basename = cmdline;
    }

    strncpy(name, basename, size - 1);
    name[size - 1] = '\0';
    return 0;
}

// Helper to check if parent is a trusted compiler
static int is_trusted_compiler_parent(void) {
    char parent_name[256];

    if (get_parent_process_name(parent_name, sizeof(parent_name)) != 0) {
        return 0; // Couldn't determine parent, deny
    }

    // Trusted compiler toolchains
    const char *trusted[] = {
        "g++", "gcc", "c++", "cc",
        "clang", "clang++",
        "ld", "ld.gold", "ld.lld",  // Allow ld calling ld (multi-stage linking)
        "collect2",  // GCC's internal linker wrapper
        NULL
    };

    for (int i = 0; trusted[i] != NULL; i++) {
        if (strcmp(parent_name, trusted[i]) == 0) {
            return 1; // Parent is trusted
        }
    }

    return 0; // Parent not in trusted list
}

// Helper to check if command is dangerous
static int is_dangerous(const char *file, char *const argv[]) {
    // ========================================================================
    // WHITELIST: Trusted Development Tools
    // ========================================================================
    // Allow Claude Code and other development tools
    if (strstr(file, "claude") || strstr(file, "node") ||
        strstr(file, "npm") || strstr(file, "yarn") || strstr(file, "pnpm") ||
        strcmp(file, "ls") == 0 || strcmp(file, "cat") == 0 ||
        strcmp(file, "grep") == 0 || strcmp(file, "find") == 0 ||
        strcmp(file, "which") == 0 || strcmp(file, "echo") == 0) {
        return 0;  // Trusted development tool or safe command
    }

    // Check all args for dangerous patterns
    for (int i = 0; argv[i] != NULL; i++) {
        const char *arg = argv[i];
        const char *cmd = (i == 0) ? file : arg;  // Check cmd name too

        // ========================================================================
        // 1. DISK/FILESYSTEM DESTRUCTION
        // ========================================================================

        // rm with dangerous flags/paths
        if (strstr(cmd, "rm")) {
            for (int j = 1; argv[j] != NULL; j++) {
                const char *flag = argv[j];
                // Check for recursive/force flags
                if (strcmp(flag, "-r") == 0 || strcmp(flag, "-f") == 0 ||
                    strcmp(flag, "-rf") == 0 || strcmp(flag, "-fr") == 0 ||
                    strcmp(flag, "--recursive") == 0 || strcmp(flag, "--force") == 0) {

                    // Check next arg for critical paths
                    const char *path = argv[j + 1];
                    if (path && (strcmp(path, "/") == 0 || strcmp(path, "~") == 0 ||
                                 strcmp(path, "/*") == 0 || strcmp(path, "/home") == 0 ||
                                 strstr(path, "/home/") || strstr(path, "/etc/") ||
                                 strstr(path, "/var/") || strstr(path, "/usr/") ||
                                 strstr(path, "/boot/") || strstr(path, "/opt/"))) {
                        return 1;  // Dangerous rm
                    }
                }
            }
        }

        // dd (disk destroyer)
        if (strstr(cmd, "dd")) {
            for (int j = 0; argv[j] != NULL; j++) {
                // Check for output to devices or dangerous input sources
                if (strstr(argv[j], "of=/dev/") || strstr(argv[j], "if=/dev/zero") ||
                    strstr(argv[j], "if=/dev/urandom")) {
                    return 1;  // Dangerous dd
                }
            }
        }

        // Filesystem formatters
        if (strstr(cmd, "mkfs") || strstr(cmd, "mke2fs") || strstr(cmd, "mkswap") ||
            strstr(cmd, "fdisk") || strstr(cmd, "parted") || strstr(cmd, "gdisk")) {
            return 1;
        }

        // Secure deletion
        if (strstr(cmd, "shred") && strstr(arg, "-r")) {
            return 1;  // Recursive shred
        }

        // Moving/copying to /dev/null
        if ((strstr(cmd, "mv") || strstr(cmd, "cp")) && strstr(arg, "/dev/null")) {
            return 1;
        }

        // ========================================================================
        // 2. PRIVILEGE ESCALATION
        // ========================================================================

        // Permission changes
        if (strstr(cmd, "chown") || strstr(cmd, "chmod") || strstr(cmd, "chgrp")) {
            for (int j = 0; argv[j] != NULL; j++) {
                // Recursive on critical paths
                if ((strcmp(argv[j], "-R") == 0 || strcmp(argv[j], "--recursive") == 0) &&
                    argv[j + 1] && (strcmp(argv[j + 1], "/") == 0 ||
                                    strstr(argv[j + 1], "/home/") ||
                                    strstr(argv[j + 1], "/etc/"))) {
                    return 1;
                }
            }
        }

        // Privilege escalation tools
        if (strstr(cmd, "sudo") || strstr(cmd, "su") || strstr(cmd, "pkexec")) {
            return 1;  // Block entirely
        }

        // User/group modifications
        if (strstr(cmd, "useradd") || strstr(cmd, "usermod") || strstr(cmd, "userdel") ||
            strstr(cmd, "passwd") || strstr(cmd, "groupadd") || strstr(cmd, "groupmod")) {
            return 1;
        }

        // ========================================================================
        // 3. RESOURCE EXHAUSTION
        // ========================================================================

        // Fork bombs and infinite loops
        if (strstr(arg, ":(){:|:&};:") || strstr(arg, "fork") ||
            strstr(arg, "while true") || strstr(arg, "while :")) {
            return 1;
        }

        // yes command abuse (disk filler)
        if (strstr(cmd, "yes") && strstr(arg, ">")) {
            return 1;
        }

        // ========================================================================
        // 4. NETWORK/EXTERNAL THREATS
        // ========================================================================

        // curl/wget piped to shell
        if ((strstr(cmd, "curl") || strstr(cmd, "wget"))) {
            for (int j = 0; argv[j] != NULL; j++) {
                if ((strstr(argv[j], "http://") || strstr(argv[j], "https://") ||
                     strstr(argv[j], "ftp://")) &&
                    (strstr(argv[j + 1], "|") || strstr(argv[j + 1], "bash") ||
                     strstr(argv[j + 1], "sh"))) {
                    return 1;  // Download and execute
                }
            }
        }

        // SSH/SCP to root
        if ((strstr(cmd, "ssh") || strstr(cmd, "scp")) && strstr(arg, "root@")) {
            return 1;
        }

        // ========================================================================
        // 5. SYSTEM DISRUPTION
        // ========================================================================

        // Reboot/shutdown
        if (strstr(cmd, "reboot") || strstr(cmd, "shutdown") ||
            strstr(cmd, "halt") || strstr(cmd, "poweroff") || strstr(cmd, "init")) {
            return 1;
        }

        // systemctl service disruption
        if (strstr(cmd, "systemctl")) {
            for (int j = 0; argv[j] != NULL; j++) {
                if (strcmp(argv[j], "stop") == 0 || strcmp(argv[j], "disable") == 0 ||
                    strcmp(argv[j], "mask") == 0) {
                    return 1;
                }
            }
        }

        // Firewall manipulation
        if (strstr(cmd, "iptables") || strstr(cmd, "ufw") || strstr(cmd, "firewall-cmd")) {
            if (strstr(arg, "-F") || strstr(arg, "disable") || strstr(arg, "--flush")) {
                return 1;
            }
        }

        // Kernel parameter changes
        if (strstr(cmd, "sysctl") && strstr(arg, "kernel.")) {
            return 1;
        }

        // ========================================================================
        // 6. OBFUSCATION TECHNIQUES
        // ========================================================================

        // base64 decode to pipe
        if (strstr(cmd, "base64") && strstr(arg, "-d")) {
            for (int j = 0; argv[j] != NULL; j++) {
                if (strstr(argv[j], "|") || strstr(argv[j + 1], "bash") ||
                    strstr(argv[j + 1], "sh")) {
                    return 1;
                }
            }
        }

        // echo to shell
        if (strstr(cmd, "echo")) {
            for (int j = 0; argv[j] != NULL; j++) {
                if (strstr(argv[j], "|") && (strstr(argv[j + 1], "bash") ||
                                             strstr(argv[j + 1], "sh"))) {
                    return 1;
                }
            }
        }

        // eval abuse
        if (strstr(cmd, "eval") || strstr(cmd, "exec")) {
            return 1;
        }

        // ========================================================================
        // 7. BACKDOORS & REVERSE SHELLS
        // ========================================================================

        // Netcat/nc reverse shells
        if (strstr(cmd, "nc") || strstr(cmd, "netcat")) {
            for (int j = 0; argv[j] != NULL; j++) {
                // Check for listening ports or connections
                if (strcmp(argv[j], "-l") == 0 || strcmp(argv[j], "-lvp") == 0 ||
                    strcmp(argv[j], "-e") == 0 || strstr(argv[j], "/bin/bash") ||
                    strstr(argv[j], "/bin/sh")) {
                    return 1;  // Reverse shell patterns
                }
            }
        }

        // Bash reverse shells
        if (strstr(arg, "bash -i") || strstr(arg, "/dev/tcp/") || strstr(arg, "/dev/udp/")) {
            return 1;  // e.g., bash -i >& /dev/tcp/attacker/4444
        }

        // Crypto miners (common process names)
        if (strstr(cmd, "xmrig") || strstr(cmd, "minerd") || strstr(cmd, "cpuminer") ||
            strstr(cmd, "ethminer") || strstr(cmd, "phoenixminer")) {
            return 1;
        }

        // ========================================================================
        // 8. ENVIRONMENT VARIABLE MANIPULATION
        // ========================================================================

        // PATH manipulation
        if (strstr(cmd, "export")) {
            for (int j = 0; argv[j] != NULL; j++) {
                if (strstr(argv[j], "PATH=") && !strstr(argv[j], "/usr/bin") &&
                    !strstr(argv[j], "/bin")) {
                    return 1;  // Suspicious PATH override
                }
            }
        }

        // History hiding
        if (strstr(arg, "unset HISTFILE") || strstr(arg, "HISTFILE=/dev/null") ||
            strstr(arg, "export HISTFILESIZE=0")) {
            return 1;
        }

        // ========================================================================
        // PROJECT WARDEN: THE SCRIBE'S PASS
        // Special handling for linker (ld) when called by trusted compilers
        // ========================================================================

        // Check if this is the linker being executed
        if (strstr(file, "/ld") || strcmp(file, "ld") == 0 ||
            strstr(file, "ld.gold") || strstr(file, "ld.lld")) {

            // Check if SAFE_EXEC_ALLOW_LINKING is enabled (Developer Mode)
            const char *allow_linking = getenv("SAFE_EXEC_ALLOW_LINKING");
            if (allow_linking && strcmp(allow_linking, "1") == 0) {
                // In Developer Mode: Allow if parent is a trusted compiler
                if (is_trusted_compiler_parent()) {
                    // The Scribe's Pass: Compiler → Linker is legitimate
                    return 0;  // Allow linking
                }
                // Parent is not trusted, fall through to other checks
            }
            // If SAFE_EXEC_ALLOW_LINKING is not set, treat ld as potentially dangerous
            // and let it be caught by LD_PRELOAD check below
        }

        // LD_PRELOAD hijacking (ironic, but block for safety)
        if (strstr(arg, "LD_PRELOAD=") && !strstr(arg, "safe_exec.so") &&
            !strstr(arg, "safe_open.so") && !strstr(arg, "safe_fork.so")) {
            return 1;  // Block unauthorized LD_PRELOAD
        }

        // ========================================================================
        // 9. KERNEL & MODULE EXPLOITS
        // ========================================================================

        // Kernel module loading
        if (strstr(cmd, "insmod") || strstr(cmd, "modprobe") || strstr(cmd, "rmmod")) {
            return 1;  // Block all module operations
        }

        // Kernel compilation/patching
        if (strstr(cmd, "make") && (strstr(arg, "bzImage") || strstr(arg, "vmlinuz"))) {
            return 1;  // Kernel builds
        }

        // Direct kernel access
        if (strstr(arg, "/dev/kmem") || strstr(arg, "/dev/mem") || strstr(arg, "/dev/port")) {
            return 1;
        }

        // ========================================================================
        // 10. SCRIPTED INJECTIONS (Python/Perl/Ruby)
        // ========================================================================

        // Python -c injections
        if (strstr(cmd, "python") || strstr(cmd, "python3") || strstr(cmd, "python2")) {
            for (int j = 0; argv[j] != NULL; j++) {
                if (strcmp(argv[j], "-c") == 0 && argv[j + 1]) {
                    const char *code = argv[j + 1];
                    // Check for dangerous os/subprocess/system calls
                    if (strstr(code, "os.system") || strstr(code, "subprocess") ||
                        strstr(code, "import os") || strstr(code, "__import__") ||
                        strstr(code, "exec(") || strstr(code, "eval(") ||
                        strstr(code, "rm -rf") || strstr(code, "/dev/")) {
                        return 1;  // Dangerous Python code
                    }
                }
            }
        }

        // Perl -e injections
        if (strstr(cmd, "perl")) {
            for (int j = 0; argv[j] != NULL; j++) {
                if (strcmp(argv[j], "-e") == 0 && argv[j + 1]) {
                    const char *code = argv[j + 1];
                    // Check for dangerous Perl system/exec calls
                    if (strstr(code, "system") || strstr(code, "exec") ||
                        strstr(code, "`") || strstr(code, "qx") ||
                        strstr(code, "rm -rf") || strstr(code, "/dev/")) {
                        return 1;  // Dangerous Perl code
                    }
                }
            }
        }

        // Ruby -e injections
        if (strstr(cmd, "ruby")) {
            for (int j = 0; argv[j] != NULL; j++) {
                if (strcmp(argv[j], "-e") == 0 && argv[j + 1]) {
                    const char *code = argv[j + 1];
                    // Check for dangerous Ruby system/exec calls
                    if (strstr(code, "system") || strstr(code, "exec") ||
                        strstr(code, "`") || strstr(code, "%x") ||
                        strstr(code, "rm -rf") || strstr(code, "/dev/")) {
                        return 1;  // Dangerous Ruby code
                    }
                }
            }
        }

        // Node.js/JavaScript injections
        if (strstr(cmd, "node") || strstr(cmd, "nodejs")) {
            for (int j = 0; argv[j] != NULL; j++) {
                if (strcmp(argv[j], "-e") == 0 && argv[j + 1]) {
                    const char *code = argv[j + 1];
                    // Check for dangerous Node.js child_process calls
                    if (strstr(code, "child_process") || strstr(code, "exec") ||
                        strstr(code, "spawn") || strstr(code, "require('fs')") ||
                        strstr(code, "rm -rf") || strstr(code, "/dev/")) {
                        return 1;  // Dangerous Node.js code
                    }
                }
            }
        }

        // ========================================================================
        // 11. PACKAGE MANAGEMENT (Optional - may have false positives)
        // ========================================================================

        // Uncomment to block package operations
        /*
        if (strstr(cmd, "apt") || strstr(cmd, "yum") || strstr(cmd, "dnf") ||
            strstr(cmd, "pacman") || strstr(cmd, "zypper")) {
            if (strstr(arg, "remove") || strstr(arg, "purge") || strstr(arg, "erase")) {
                return 1;  // Block package removal
            }
        }
        */
    }

    return 0;  // Safe command
}

// Intercepted execvp
int execvp(const char *file, char *const argv[]) {
    // Lazy-load original execvp
    if (!original_execvp) {
        original_execvp = dlsym(RTLD_NEXT, "execvp");
        if (!original_execvp) {
            fprintf(stderr, "[SAFE_EXEC] Error: Failed to load original execvp\n");
            errno = ENOENT;
            return -1;
        }
    }

    // Check if logging is enabled via env var
    const char *log_enabled = getenv("SAFE_EXEC_LOG");
    int should_log = log_enabled && (strcmp(log_enabled, "1") == 0 || strcmp(log_enabled, "true") == 0);

    // Check if dangerous
    if (is_dangerous(file, argv)) {
        fprintf(stderr, "\n[SAFE_EXEC] ⛔ BLOCKED DANGEROUS COMMAND:\n[SAFE_EXEC] ");
        for (int i = 0; argv[i] != NULL; i++) {
            fprintf(stderr, "%s ", argv[i]);
        }
        fprintf(stderr, "\n[SAFE_EXEC] Reason: Pattern matched threat database\n");
        fprintf(stderr, "[SAFE_EXEC] Set SAFE_EXEC_OVERRIDE=1 to bypass (NOT RECOMMENDED)\n\n");

        // Check for override (emergency escape hatch)
        const char *override = getenv("SAFE_EXEC_OVERRIDE");
        if (override && strcmp(override, "1") == 0) {
            fprintf(stderr, "[SAFE_EXEC] ⚠️  WARNING: Override enabled, executing anyway...\n");
            return original_execvp(file, argv);
        }

        errno = EPERM;  // Permission denied
        return -1;
    }

    // Log safe commands if enabled
    if (should_log) {
        fprintf(stderr, "[SAFE_EXEC] ✓ Allowed: %s", file);
        for (int i = 1; argv[i] != NULL; i++) {
            fprintf(stderr, " %s", argv[i]);
        }
        fprintf(stderr, "\n");
    }

    // Safe: Call original
    return original_execvp(file, argv);
}
