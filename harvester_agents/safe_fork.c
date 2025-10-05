#define _GNU_SOURCE
#include <dlfcn.h>       // For dlsym
#include <errno.h>       // For error codes
#include <stdio.h>       // For fprintf
#include <stdlib.h>      // For getenv
#include <string.h>      // For strcmp
#include <sys/time.h>    // For gettimeofday
#include <unistd.h>      // For fork, vfork, pid_t

// Original function pointers
static pid_t (*original_fork)(void) = NULL;
static pid_t (*original_vfork)(void) = NULL;

// Fork tracking state
static struct timeval last_fork_time = {0, 0};
static struct timeval last_activity_time = {0, 0};  // For idle reset
static int fork_count_current_second = 0;
static int total_fork_count = 0;

// Configuration (via env vars)
static int max_forks_per_second = 10;   // Default: 10 forks/sec
static int max_total_forks = 1000;      // Default: 1000 total forks
static int idle_reset_seconds = 5;      // Reset after 5 sec of no forks
static int initialized = 0;

// Initialize configuration from environment
static void init_config(void) {
    if (initialized) return;

    const char *max_per_sec = getenv("SAFE_FORK_MAX_PER_SEC");
    if (max_per_sec) {
        max_forks_per_second = atoi(max_per_sec);
        if (max_forks_per_second <= 0) max_forks_per_second = 10;
    }

    const char *max_total = getenv("SAFE_FORK_MAX_TOTAL");
    if (max_total) {
        max_total_forks = atoi(max_total);
        if (max_total_forks <= 0) max_total_forks = 1000;
    }

    const char *reset_time = getenv("SAFE_FORK_IDLE_RESET");
    if (reset_time) {
        idle_reset_seconds = atoi(reset_time);
        if (idle_reset_seconds <= 0) idle_reset_seconds = 5;
    }

    initialized = 1;

    fprintf(stderr, "[SAFE_FORK] Initialized: max %d forks/sec, %d total, %d sec idle reset\n",
            max_forks_per_second, max_total_forks, idle_reset_seconds);
}

// Check if fork should be allowed
static int should_allow_fork(void) {
    init_config();

    struct timeval now;
    gettimeofday(&now, NULL);

    // Check for idle period - reset rate counter if no activity for X seconds
    if (last_activity_time.tv_sec != 0) {
        long idle_time = now.tv_sec - last_activity_time.tv_sec;
        if (idle_time >= idle_reset_seconds) {
            // Been idle for configured period - reset rate counter
            fork_count_current_second = 0;
            last_fork_time = now;

            const char *log_enabled = getenv("SAFE_FORK_LOG");
            if (log_enabled && strcmp(log_enabled, "1") == 0) {
                fprintf(stderr, "[SAFE_FORK] ℹ️  Rate counter reset after %ld sec idle\n", idle_time);
            }
        }
    }

    // Update activity time
    last_activity_time = now;

    // Reset counter if we're in a new second
    if (now.tv_sec != last_fork_time.tv_sec) {
        fork_count_current_second = 0;
        last_fork_time = now;
    }

    // Check rate limit (forks per second)
    fork_count_current_second++;
    if (fork_count_current_second > max_forks_per_second) {
        fprintf(stderr, "\n[SAFE_FORK] ⛔ BLOCKED: Fork rate limit exceeded\n");
        fprintf(stderr, "[SAFE_FORK] Current: %d forks/sec, limit: %d\n",
                fork_count_current_second, max_forks_per_second);
        fprintf(stderr, "[SAFE_FORK] Possible fork bomb detected!\n");
        fprintf(stderr, "[SAFE_FORK] Set SAFE_FORK_OVERRIDE=1 to bypass\n\n");

        // Check for override
        const char *override = getenv("SAFE_FORK_OVERRIDE");
        if (override && strcmp(override, "1") == 0) {
            fprintf(stderr, "[SAFE_FORK] ⚠️  WARNING: Override enabled, allowing anyway...\n");
            return 1;
        }

        return 0;
    }

    // Check total fork limit (lifetime)
    total_fork_count++;
    if (total_fork_count > max_total_forks) {
        fprintf(stderr, "\n[SAFE_FORK] ⛔ BLOCKED: Total fork limit exceeded\n");
        fprintf(stderr, "[SAFE_FORK] Total forks: %d, limit: %d\n",
                total_fork_count, max_total_forks);
        fprintf(stderr, "[SAFE_FORK] This prevents resource exhaustion attacks\n");
        fprintf(stderr, "[SAFE_FORK] Set SAFE_FORK_OVERRIDE=1 to bypass\n\n");

        // Check for override
        const char *override = getenv("SAFE_FORK_OVERRIDE");
        if (override && strcmp(override, "1") == 0) {
            fprintf(stderr, "[SAFE_FORK] ⚠️  WARNING: Override enabled, allowing anyway...\n");
            return 1;
        }

        return 0;
    }

    // Check for logging
    const char *log_enabled = getenv("SAFE_FORK_LOG");
    if (log_enabled && strcmp(log_enabled, "1") == 0) {
        fprintf(stderr, "[SAFE_FORK] ✓ Allowed fork #%d (rate: %d/sec)\n",
                total_fork_count, fork_count_current_second);
    }

    return 1;
}

// Intercepted fork
pid_t fork(void) {
    // Lazy-load original fork
    if (!original_fork) {
        original_fork = dlsym(RTLD_NEXT, "fork");
        if (!original_fork) {
            fprintf(stderr, "[SAFE_FORK] Error: Failed to load original fork\n");
            errno = ENOSYS;
            return -1;
        }
    }

    // Check if allowed
    if (!should_allow_fork()) {
        errno = EAGAIN;  // Resource temporarily unavailable
        return -1;
    }

    return original_fork();
}

// Intercepted vfork (similar protection)
pid_t vfork(void) {
    // Lazy-load original vfork
    if (!original_vfork) {
        original_vfork = dlsym(RTLD_NEXT, "vfork");
        if (!original_vfork) {
            fprintf(stderr, "[SAFE_FORK] Error: Failed to load original vfork\n");
            errno = ENOSYS;
            return -1;
        }
    }

    // Check if allowed (same logic as fork)
    if (!should_allow_fork()) {
        errno = EAGAIN;
        return -1;
    }

    return original_vfork();
}
