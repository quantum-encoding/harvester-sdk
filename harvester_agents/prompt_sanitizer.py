"""
Prompt Sanitization Module
Defends against Unicode smuggling, emoji injection, and other obfuscation techniques
"""

import re
import unicodedata
from typing import Optional
from urllib.parse import urlparse

# Try to import ftfy for advanced Unicode fixing
try:
    import ftfy
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False


class PromptSanitizer:
    """Sanitize prompts to prevent injection attacks via Unicode and link smuggling."""

    # Dangerous URL schemes
    DANGEROUS_SCHEMES = {
        'data', 'javascript', 'blob', 'file', 'vbscript',
        'about', 'chrome', 'chrome-extension'
    }

    # Known malicious patterns (regex)
    MALICIOUS_PATTERNS = [
        # Base64-like strings (potential encoded commands)
        r'[A-Za-z0-9+/]{30,}={0,2}',
        # Hexadecimal shells
        r'\\x[0-9a-fA-F]{2}',
        # Unicode escape sequences
        r'\\u[0-9a-fA-F]{4}',
        # Suspicious command patterns
        r'rm\s+-rf\s+/',
        r'dd\s+if=',
        r'mkfs',
        r'\|\s*(bash|sh)',
        r'curl.*\|',
        r'wget.*\|',
    ]

    @staticmethod
    def normalize_unicode(text: str, aggressive: bool = True, use_ftfy: bool = True) -> str:
        """Normalize Unicode to prevent smuggling attacks.

        Args:
            text: Input text that may contain Unicode tricks
            aggressive: If True, strips all non-ASCII. If False, only normalizes.
            use_ftfy: If True and ftfy is installed, use it for deeper fixing

        Returns:
            Sanitized text
        """
        # Step 0: Advanced Unicode fixing with ftfy (if available)
        if use_ftfy and HAS_FTFY:
            # ftfy fixes mojibake, mixed encodings, and other Unicode corruption
            # e.g., "Ã©" → "é", broken surrogate pairs, etc.
            text = ftfy.fix_text(text)

        # Step 1: Unicode normalization (NFKD = compatibility decomposition)
        # This converts fancy Unicode variants to their base ASCII equivalents
        # e.g., "ｒｍ" (fullwidth) → "rm", "𝗿𝗺" (bold) → "rm"
        normalized = unicodedata.normalize('NFKD', text)

        if aggressive:
            # Step 2: Strip all non-ASCII characters
            # This removes emojis, zero-width chars, and other smuggling vectors
            ascii_only = normalized.encode('ascii', 'ignore').decode('ascii')
            return ascii_only
        else:
            # Less aggressive: keep valid Unicode but remove control chars
            return ''.join(char for char in normalized
                          if not unicodedata.category(char).startswith('C'))

    @staticmethod
    def remove_zero_width_chars(text: str) -> str:
        """Remove zero-width and invisible characters used for smuggling.

        Common smuggling chars:
        - U+200B (Zero Width Space)
        - U+200C (Zero Width Non-Joiner)
        - U+200D (Zero Width Joiner)
        - U+FEFF (Zero Width No-Break Space / BOM)
        - U+180E (Mongolian Vowel Separator)
        """
        zero_width_chars = [
            '\u200B',  # ZWSP
            '\u200C',  # ZWNJ
            '\u200D',  # ZWJ
            '\uFEFF',  # ZWNBSP
            '\u180E',  # MVS
            '\u2060',  # Word Joiner
            '\u2061',  # Function Application
            '\u2062',  # Invisible Times
            '\u2063',  # Invisible Separator
            '\u2064',  # Invisible Plus
        ]

        for char in zero_width_chars:
            text = text.replace(char, '')

        return text

    @staticmethod
    def strip_emojis(text: str) -> str:
        """Remove all emojis from text.

        This prevents emoji-based smuggling where commands are hidden
        in emoji sequences or emoji metadata.
        """
        # Emoji ranges (common blocks)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & pictographs
            "\U0001F680-\U0001F6FF"  # Transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub('', text)

    @staticmethod
    def validate_urls(text: str, allow_http: bool = True) -> tuple[str, list[str]]:
        """Extract and validate URLs, blocking dangerous schemes.

        Args:
            text: Input text
            allow_http: Whether to allow http/https URLs

        Returns:
            Tuple of (sanitized_text, list_of_blocked_urls)
        """
        blocked_urls = []

        # Find all URLs
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        urls = re.findall(url_pattern, text)

        for url in urls:
            try:
                parsed = urlparse(url)
                scheme = parsed.scheme.lower()

                # Block dangerous schemes
                if scheme in PromptSanitizer.DANGEROUS_SCHEMES:
                    blocked_urls.append(url)
                    # Replace with placeholder
                    text = text.replace(url, '[BLOCKED_URL]')

                # Optionally block http/https
                elif not allow_http and scheme in ('http', 'https'):
                    blocked_urls.append(url)
                    text = text.replace(url, '[BLOCKED_URL]')

            except Exception:
                # If parsing fails, assume dangerous
                blocked_urls.append(url)
                text = text.replace(url, '[BLOCKED_URL]')

        return text, blocked_urls

    @staticmethod
    def detect_malicious_patterns(text: str) -> list[str]:
        """Detect known malicious patterns in text.

        Returns:
            List of matched patterns (for logging/alerting)
        """
        matches = []

        for pattern in PromptSanitizer.MALICIOUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)

        return matches

    @classmethod
    def sanitize(cls, prompt: str, aggressive: bool = True,
                 allow_http: bool = False) -> dict:
        """Full sanitization pipeline.

        Args:
            prompt: Raw prompt from user/agent
            aggressive: Use aggressive Unicode stripping
            allow_http: Allow http/https URLs

        Returns:
            Dict with:
                - sanitized: Cleaned prompt
                - blocked_urls: List of blocked URLs
                - detected_patterns: List of suspicious patterns
                - safe: Boolean indicating if prompt seems safe
        """
        original = prompt

        # Step 1: Remove zero-width smuggling chars
        prompt = cls.remove_zero_width_chars(prompt)

        # Step 2: Strip emojis
        prompt = cls.strip_emojis(prompt)

        # Step 3: Normalize Unicode
        prompt = cls.normalize_unicode(prompt, aggressive=aggressive)

        # Step 4: Validate URLs
        prompt, blocked_urls = cls.validate_urls(prompt, allow_http=allow_http)

        # Step 5: Detect malicious patterns
        detected_patterns = cls.detect_malicious_patterns(prompt)

        # Determine if safe
        is_safe = len(blocked_urls) == 0 and len(detected_patterns) == 0

        return {
            'original': original,
            'sanitized': prompt,
            'blocked_urls': blocked_urls,
            'detected_patterns': detected_patterns,
            'safe': is_safe,
            'changes_made': original != prompt,
        }


def sanitize_prompt(prompt: str, **kwargs) -> str:
    """Convenience function for quick sanitization.

    Returns only the sanitized text. For full analysis, use
    PromptSanitizer.sanitize() directly.
    """
    result = PromptSanitizer.sanitize(prompt, **kwargs)
    return result['sanitized']


# Example usage and tests
if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Unicode smuggling
        "Run this: ｒｍ -rf /",  # Fullwidth chars
        "Execute: 𝗿𝗺 -rf /",   # Math bold
        "Try: rm\u200B-rf\u200C/",  # Zero-width chars

        # Emoji smuggling
        "😊 Help with: rm -rf / 😊",

        # Base64 injection
        "Decode this: cm0gLXJmIC8=",

        # Link smuggling
        "Visit: javascript:alert(document.cookie)",
        "Check: data:text/html,<script>alert(1)</script>",

        # Safe prompts
        "What is the weather today?",
        "Help me write a Python function",
    ]

    print("Prompt Sanitization Test Results\n" + "="*60)

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test[:50]}...")
        result = PromptSanitizer.sanitize(test, aggressive=True, allow_http=False)

        print(f"  Original: {result['original'][:60]}")
        print(f"  Sanitized: {result['sanitized'][:60]}")
        print(f"  Safe: {result['safe']}")

        if result['blocked_urls']:
            print(f"  Blocked URLs: {result['blocked_urls']}")
        if result['detected_patterns']:
            print(f"  Detected patterns: {len(result['detected_patterns'])}")
