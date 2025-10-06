#define _GNU_SOURCE
#include <dlfcn.h>     // For dlsym
#include <errno.h>     // For error codes
#include <fcntl.h>     // For open, O_WRONLY, etc.
#include <stdarg.h>    // For va_list, va_start, va_end
#include <stdio.h>     // For fprintf, FILE, fopen
#include <string.h>    // For strstr, strcmp
#include <stdlib.h>    // For getenv

// Original function pointers
static int (*original_open)(const char *pathname, int flags, ...) = NULL;
static FILE *(*original_fopen)(const char *pathname, const char *mode) = NULL;

// Helper to check if path is dangerous to write to
static int is_dangerous_path(const char *path, int is_write) {
    if (!is_write) {
        return 0;  // Read operations are generally safe
    }

    // Critical system paths that should never be written to
    if (strcmp(path, "/dev/sda") == 0 || strcmp(path, "/dev/vda") == 0 ||
        strcmp(path, "/dev/nvme0n1") == 0 || strcmp(path, "/dev/hda") == 0) {
        return 1;  // Block writing to root disk devices
    }

    // Block writes to critical directories
    if (strstr(path, "/etc/") == path || strcmp(path, "/etc") == 0) {
        return 1;  // System configuration
    }

    if (strstr(path, "/boot/") == path || strcmp(path, "/boot") == 0) {
        return 1;  // Boot partition
    }

    if (strstr(path, "/sys/") == path || strcmp(path, "/sys") == 0) {
        return 1;  // Kernel sysfs
    }

    // Allow writes to /proc/self/* (process-specific, safe)
    if (strstr(path, "/proc/self/") == path) {
        return 0;  // Safe - process writing to its own proc info
    }

    if (strstr(path, "/proc/") == path || strcmp(path, "/proc") == 0) {
        return 1;  // Process filesystem (some writes are dangerous)
    }

    // Specific dangerous /proc/ writes (kernel parameters)
    if (strstr(path, "/proc/sys/kernel/") || strstr(path, "/proc/sys/vm/") ||
        strstr(path, "/proc/sys/net/") || strstr(path, "/proc/sysrq-trigger")) {
        return 1;  // Kernel parameter manipulation
    }

    // Block writes to critical system files
    if (strcmp(path, "/etc/passwd") == 0 || strcmp(path, "/etc/shadow") == 0 ||
        strcmp(path, "/etc/sudoers") == 0 || strcmp(path, "/etc/group") == 0) {
        return 1;  // User/authentication files
    }

    if (strcmp(path, "/etc/fstab") == 0 || strcmp(path, "/etc/hosts") == 0 ||
        strcmp(path, "/etc/resolv.conf") == 0) {
        return 1;  // Critical system config files
    }

    // Block writes to init systems
    if (strstr(path, "/etc/systemd/") || strstr(path, "/etc/init.d/") ||
        strstr(path, "/lib/systemd/")) {
        return 1;
    }

    // Check for override
    const char *override = getenv("SAFE_OPEN_OVERRIDE");
    if (override && strcmp(override, "1") == 0) {
        return 0;  // Override enabled, allow
    }

    return 0;  // Safe path
}

// Intercepted open
int open(const char *pathname, int flags, ...) {
    // Lazy-load original open
    if (!original_open) {
        original_open = dlsym(RTLD_NEXT, "open");
        if (!original_open) {
            fprintf(stderr, "[SAFE_OPEN] Error: Failed to load original open\n");
            errno = ENOENT;
            return -1;
        }
    }

    // Check if write operation
    int is_write = (flags & (O_WRONLY | O_RDWR | O_CREAT | O_TRUNC));

    // Check if path is dangerous
    if (is_dangerous_path(pathname, is_write)) {
        fprintf(stderr, "\n[SAFE_OPEN] ⛔ BLOCKED DANGEROUS FILE WRITE:\n");
        fprintf(stderr, "[SAFE_OPEN] Path: %s\n", pathname);
        fprintf(stderr, "[SAFE_OPEN] Flags: %d (write mode: %s)\n", flags, is_write ? "yes" : "no");
        fprintf(stderr, "[SAFE_OPEN] Reason: Writing to critical system path\n");
        fprintf(stderr, "[SAFE_OPEN] Set SAFE_OPEN_OVERRIDE=1 to bypass (NOT RECOMMENDED)\n\n");
        errno = EACCES;  // Permission denied
        return -1;
    }

    // Extract mode if O_CREAT is set
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list args;
        va_start(args, flags);
        mode = va_arg(args, mode_t);
        va_end(args);
        return original_open(pathname, flags, mode);
    }

    return original_open(pathname, flags);
}

// Intercepted fopen
FILE *fopen(const char *pathname, const char *mode) {
    // Lazy-load original fopen
    if (!original_fopen) {
        original_fopen = dlsym(RTLD_NEXT, "fopen");
        if (!original_fopen) {
            fprintf(stderr, "[SAFE_OPEN] Error: Failed to load original fopen\n");
            errno = ENOENT;
            return NULL;
        }
    }

    // Check if write mode (w, w+, a, a+, r+)
    int is_write = (strchr(mode, 'w') != NULL || strchr(mode, 'a') != NULL ||
                    strchr(mode, '+') != NULL);

    // Check if path is dangerous
    if (is_dangerous_path(pathname, is_write)) {
        fprintf(stderr, "\n[SAFE_OPEN] ⛔ BLOCKED DANGEROUS FILE WRITE:\n");
        fprintf(stderr, "[SAFE_OPEN] Path: %s\n", pathname);
        fprintf(stderr, "[SAFE_OPEN] Mode: %s (write mode: %s)\n", mode, is_write ? "yes" : "no");
        fprintf(stderr, "[SAFE_OPEN] Reason: Writing to critical system path\n");
        fprintf(stderr, "[SAFE_OPEN] Set SAFE_OPEN_OVERRIDE=1 to bypass (NOT RECOMMENDED)\n\n");
        errno = EACCES;  // Permission denied
        return NULL;
    }

    // Check for logging
    const char *log_enabled = getenv("SAFE_OPEN_LOG");
    if (log_enabled && strcmp(log_enabled, "1") == 0) {
        fprintf(stderr, "[SAFE_OPEN] ✓ Allowed: fopen(%s, %s)\n", pathname, mode);
    }

    return original_fopen(pathname, mode);
}
