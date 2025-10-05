#!/usr/bin/env python3
"""
Security Log Monitor
Analyzes logs from safe_exec/open/fork to detect attack patterns and anomalies
"""

import re
import sys
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Optional


class SecurityMonitor:
    """Monitor and analyze security library logs."""

    def __init__(self, log_file: Optional[str] = None):
        """Initialize monitor.

        Args:
            log_file: Path to log file. If None, reads from stdin.
        """
        self.log_file = log_file
        self.events = []
        self.statistics = defaultdict(int)
        self.blocked_commands = []
        self.blocked_files = []
        self.fork_violations = []

    def parse_logs(self):
        """Parse logs from file or stdin."""
        if self.log_file:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
        else:
            print("Reading from stdin (paste logs, Ctrl+D when done)...")
            lines = sys.stdin.readlines()

        for line in lines:
            self._parse_line(line.strip())

    def _parse_line(self, line: str):
        """Parse a single log line."""
        if not line:
            return

        # SAFE_EXEC events
        if '[SAFE_EXEC]' in line:
            if 'BLOCKED' in line:
                self.statistics['exec_blocked'] += 1
                # Extract command
                match = re.search(r'BLOCKED.*:\s*(.+)', line)
                if match:
                    self.blocked_commands.append({
                        'timestamp': datetime.now(),
                        'command': match.group(1),
                        'line': line
                    })
            elif 'Allowed' in line:
                self.statistics['exec_allowed'] += 1

        # SAFE_OPEN events
        if '[SAFE_OPEN]' in line:
            if 'BLOCKED' in line:
                self.statistics['open_blocked'] += 1
                # Extract path
                match = re.search(r'Path:\s*(.+)', line)
                if match:
                    self.blocked_files.append({
                        'timestamp': datetime.now(),
                        'path': match.group(1),
                        'line': line
                    })
            elif 'Allowed' in line:
                self.statistics['open_allowed'] += 1

        # SAFE_FORK events
        if '[SAFE_FORK]' in line:
            if 'BLOCKED' in line:
                self.statistics['fork_blocked'] += 1
                self.fork_violations.append({
                    'timestamp': datetime.now(),
                    'reason': 'rate_limit' if 'rate limit' in line else 'total_limit',
                    'line': line
                })
            elif 'Allowed' in line:
                self.statistics['fork_allowed'] += 1

    def detect_attack_patterns(self):
        """Analyze logs for attack patterns."""
        patterns = {
            'disk_destruction': 0,
            'privilege_escalation': 0,
            'resource_exhaustion': 0,
            'network_threats': 0,
            'obfuscation': 0,
            'backdoors': 0,
            'scripted_injection': 0,
        }

        # Keywords for each category
        keywords = {
            'disk_destruction': ['rm -rf', 'dd', 'mkfs', 'shred', 'fdisk'],
            'privilege_escalation': ['sudo', 'su', 'chmod', 'chown', 'useradd', 'passwd'],
            'resource_exhaustion': ['fork bomb', ':(){', 'while true', 'yes >'],
            'network_threats': ['curl', 'wget', 'nc -', 'bash -i', '/dev/tcp'],
            'obfuscation': ['base64', 'echo |', 'eval', 'exec'],
            'backdoors': ['xmrig', 'minerd', 'insmod', '/dev/kmem'],
            'scripted_injection': ['python -c', 'perl -e', 'ruby -e', 'node -e'],
        }

        for blocked in self.blocked_commands:
            cmd = blocked['command'].lower()
            for category, words in keywords.items():
                if any(word.lower() in cmd for word in words):
                    patterns[category] += 1

        return patterns

    def generate_report(self):
        """Generate security report."""
        report = []
        report.append("="*80)
        report.append("SECURITY MONITOR REPORT")
        report.append("="*80)
        report.append("")

        # Summary statistics
        report.append("SUMMARY")
        report.append("-"*80)
        report.append(f"Total events analyzed: {sum(self.statistics.values())}")
        report.append("")
        report.append(f"Execution attempts:")
        report.append(f"  Blocked: {self.statistics['exec_blocked']}")
        report.append(f"  Allowed: {self.statistics['exec_allowed']}")
        report.append("")
        report.append(f"File write attempts:")
        report.append(f"  Blocked: {self.statistics['open_blocked']}")
        report.append(f"  Allowed: {self.statistics['open_allowed']}")
        report.append("")
        report.append(f"Fork attempts:")
        report.append(f"  Blocked: {self.statistics['fork_blocked']}")
        report.append(f"  Allowed: {self.statistics['fork_allowed']}")
        report.append("")

        # Attack patterns
        if self.blocked_commands:
            patterns = self.detect_attack_patterns()
            report.append("ATTACK PATTERNS DETECTED")
            report.append("-"*80)
            for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    report.append(f"  {pattern.replace('_', ' ').title()}: {count} attempts")
            report.append("")

        # Top blocked commands
        if self.blocked_commands:
            report.append("TOP BLOCKED COMMANDS")
            report.append("-"*80)
            cmd_counter = Counter(b['command'] for b in self.blocked_commands)
            for cmd, count in cmd_counter.most_common(10):
                report.append(f"  [{count}x] {cmd[:70]}")
            report.append("")

        # Top blocked files
        if self.blocked_files:
            report.append("TOP BLOCKED FILE WRITES")
            report.append("-"*80)
            file_counter = Counter(b['path'] for b in self.blocked_files)
            for path, count in file_counter.most_common(10):
                report.append(f"  [{count}x] {path}")
            report.append("")

        # Fork violations
        if self.fork_violations:
            report.append("FORK VIOLATIONS")
            report.append("-"*80)
            rate_limit = sum(1 for f in self.fork_violations if f['reason'] == 'rate_limit')
            total_limit = sum(1 for f in self.fork_violations if f['reason'] == 'total_limit')
            report.append(f"  Rate limit exceeded: {rate_limit}")
            report.append(f"  Total limit exceeded: {total_limit}")
            report.append("")

        # Security score
        total_blocked = (self.statistics['exec_blocked'] +
                        self.statistics['open_blocked'] +
                        self.statistics['fork_blocked'])
        total_allowed = (self.statistics['exec_allowed'] +
                        self.statistics['open_allowed'] +
                        self.statistics['fork_allowed'])

        report.append("SECURITY SCORE")
        report.append("-"*80)
        if total_blocked > 0:
            block_rate = (total_blocked / (total_blocked + total_allowed)) * 100 if (total_blocked + total_allowed) > 0 else 0
            report.append(f"  Attack attempts blocked: {total_blocked}")
            report.append(f"  Block rate: {block_rate:.1f}%")

            if total_blocked > 10:
                report.append("  ⚠️  HIGH THREAT LEVEL - System under active attack!")
            elif total_blocked > 3:
                report.append("  ⚠️  MODERATE THREAT - Multiple attack attempts detected")
            else:
                report.append("  ✓ LOW THREAT - Protection working as expected")
        else:
            report.append("  ✓ NO THREATS DETECTED")

        report.append("")
        report.append("="*80)

        return "\n".join(report)

    def export_json(self):
        """Export data as JSON for further analysis."""
        import json
        return json.dumps({
            'statistics': dict(self.statistics),
            'blocked_commands': [
                {
                    'timestamp': b['timestamp'].isoformat(),
                    'command': b['command']
                }
                for b in self.blocked_commands
            ],
            'blocked_files': [
                {
                    'timestamp': b['timestamp'].isoformat(),
                    'path': b['path']
                }
                for b in self.blocked_files
            ],
            'fork_violations': [
                {
                    'timestamp': f['timestamp'].isoformat(),
                    'reason': f['reason']
                }
                for f in self.fork_violations
            ],
            'attack_patterns': self.detect_attack_patterns(),
        }, indent=2)


def main():
    """CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Monitor and analyze security library logs"
    )
    parser.add_argument(
        'logfile',
        nargs='?',
        help='Log file to analyze (or stdin if not provided)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    args = parser.parse_args()

    monitor = SecurityMonitor(args.logfile)
    monitor.parse_logs()

    if args.json:
        print(monitor.export_json())
    else:
        print(monitor.generate_report())


if __name__ == "__main__":
    main()
