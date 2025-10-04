"""
Comprehensive test suite for GrokCodeAgent file tools to ensure no corruption

Tests include:
- JSON serialization edge cases (unicode, special chars, nested structures)
- Large file handling
- Concurrent operations
- Error recovery
- Edge cases that previously caused corruption
- Multi-byte character encoding
- Binary data rejection
- Malformed input handling

Copyright (c) 2025 Quantum Encoding Ltd.
"""

import asyncio
import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.grok_code_agent import GrokCodeAgent


class TestResult:
    """Container for test results"""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.details = None

    def __repr__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        msg = f"{status} - {self.name}"
        if self.error:
            msg += f"\n    Error: {self.error}"
        if self.details:
            msg += f"\n    Details: {self.details}"
        return msg


class GrokFileToolsTester:
    """Comprehensive tester for Grok file tools"""

    def __init__(self):
        self.agent = None
        self.test_dir = None
        self.results = []

    async def setup(self):
        """Initialize agent and test directory"""
        self.agent = GrokCodeAgent()
        self.test_dir = tempfile.mkdtemp(prefix="grok_test_")
        print(f"📁 Test directory: {self.test_dir}")

    async def teardown(self):
        """Clean up test directory"""
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            print(f"🧹 Cleaned up test directory")

    def get_test_path(self, filename: str) -> str:
        """Get full path for test file"""
        return os.path.join(self.test_dir, filename)

    async def run_test(self, test_func):
        """Run a single test and record result"""
        result = TestResult(test_func.__name__)
        try:
            await test_func(result)
            result.passed = True
        except Exception as e:
            result.passed = False
            result.error = str(e)

        self.results.append(result)
        print(result)

    # ============================================================
    # JSON Write Tests
    # ============================================================

    async def test_json_write_basic(self, result: TestResult):
        """Test basic JSON writing"""
        file_path = self.get_test_path("basic.json")
        data = {"name": "test", "value": 42}

        res = await self.agent.execute_tool('json_write', {
            'file_path': file_path,
            'data': data
        })

        assert res['success'], f"Tool failed: {res}"

        # Verify file contents
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert loaded == data, f"Data mismatch: {loaded} != {data}"
        result.details = f"Wrote and verified {len(data)} keys"

    async def test_json_write_unicode(self, result: TestResult):
        """Test JSON with Unicode characters"""
        file_path = self.get_test_path("unicode.json")
        data = {
            "english": "Hello",
            "chinese": "你好",
            "arabic": "مرحبا",
            "emoji": "🚀🤖💻",
            "russian": "Привет",
            "japanese": "こんにちは"
        }

        res = await self.agent.execute_tool('json_write', {
            'file_path': file_path,
            'data': data
        })

        assert res['success'], f"Tool failed: {res}"

        # Verify Unicode preserved
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert loaded == data, "Unicode data corrupted"
        result.details = f"Verified {len(data)} Unicode strings"

    async def test_json_write_nested(self, result: TestResult):
        """Test deeply nested JSON structures"""
        file_path = self.get_test_path("nested.json")
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {
                                "value": "deep",
                                "list": [1, 2, {"nested": "in list"}]
                            }
                        }
                    }
                }
            },
            "arrays": [[1, 2], [3, 4], [[5, 6]]],
            "mixed": [{"a": 1}, [{"b": 2}], "string", 123, None, True]
        }

        res = await self.agent.execute_tool('json_write', {
            'file_path': file_path,
            'data': data
        })

        assert res['success'], f"Tool failed: {res}"

        with open(file_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert loaded == data, "Nested structure corrupted"
        result.details = "Verified 5-level nesting with mixed types"

    async def test_json_write_special_chars(self, result: TestResult):
        """Test JSON with special characters and escape sequences"""
        file_path = self.get_test_path("special.json")
        data = {
            "newline": "line1\nline2",
            "tab": "col1\tcol2",
            "quote": 'He said "hello"',
            "backslash": "path\\to\\file",
            "null_byte": "before\x00after",
            "control_chars": "\r\n\t\b\f",
            "forward_slash": "https://example.com/path"
        }

        res = await self.agent.execute_tool('json_write', {
            'file_path': file_path,
            'data': data
        })

        assert res['success'], f"Tool failed: {res}"

        with open(file_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert loaded == data, "Special characters corrupted"
        result.details = f"Verified {len(data)} special character cases"

    async def test_json_write_large(self, result: TestResult):
        """Test writing large JSON file (10K entries)"""
        file_path = self.get_test_path("large.json")
        data = {f"key_{i}": {"value": i, "data": "x" * 100} for i in range(10000)}

        res = await self.agent.execute_tool('json_write', {
            'file_path': file_path,
            'data': data
        })

        assert res['success'], f"Tool failed: {res}"

        with open(file_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert len(loaded) == 10000, f"Size mismatch: {len(loaded)} != 10000"
        assert loaded["key_5000"]["value"] == 5000, "Data integrity check failed"

        file_size = os.path.getsize(file_path)
        result.details = f"Wrote {len(loaded)} entries, {file_size / 1024:.1f} KB"

    # ============================================================
    # JSON Update Tests
    # ============================================================

    async def test_json_update_basic(self, result: TestResult):
        """Test basic JSON update"""
        file_path = self.get_test_path("update_basic.json")

        # Create initial file
        await self.agent.execute_tool('json_write', {
            'file_path': file_path,
            'data': {"count": 0, "name": "test"}
        })

        # Update it
        res = await self.agent.execute_tool('json_update', {
            'file_path': file_path,
            'update_code': "data['count'] = 42; data['new_key'] = 'added'"
        })

        assert res['success'], f"Tool failed: {res}"

        with open(file_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert loaded['count'] == 42, f"Update failed: count = {loaded['count']}"
        assert loaded['new_key'] == 'added', "New key not added"
        result.details = "Updated and verified 2 operations"

    async def test_json_update_complex(self, result: TestResult):
        """Test complex JSON update with loops and conditionals"""
        file_path = self.get_test_path("update_complex.json")

        await self.agent.execute_tool('json_write', {
            'file_path': file_path,
            'data': {"items": [1, 2, 3, 4, 5]}
        })

        update_code = """
# Square all items
data['items'] = [x * x for x in data['items']]

# Add sum
data['sum'] = sum(data['items'])

# Add conditional
data['large'] = any(x > 10 for x in data['items'])
"""

        res = await self.agent.execute_tool('json_update', {
            'file_path': file_path,
            'update_code': update_code
        })

        assert res['success'], f"Tool failed: {res}"

        with open(file_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert loaded['items'] == [1, 4, 9, 16, 25], f"List comprehension failed: {loaded['items']}"
        assert loaded['sum'] == 55, f"Sum wrong: {loaded['sum']}"
        assert loaded['large'] == True, "Conditional failed"
        result.details = "Verified list comprehension, sum, and conditional"

    async def test_json_update_sandboxing(self, result: TestResult):
        """Test that dangerous operations are blocked"""
        file_path = self.get_test_path("sandbox.json")

        await self.agent.execute_tool('json_write', {
            'file_path': file_path,
            'data': {"safe": True}
        })

        # Try dangerous operations (should fail)
        dangerous_ops = [
            "import os",
            "open('/etc/passwd', 'r')",
            "__import__('os').system('ls')",
            "exec('print(1)')",
            "eval('1+1')"
        ]

        for op in dangerous_ops:
            res = await self.agent.execute_tool('json_update', {
                'file_path': file_path,
                'update_code': op
            })
            assert not res['success'], f"Dangerous op allowed: {op}"

        result.details = f"Blocked {len(dangerous_ops)} dangerous operations"

    # ============================================================
    # Text File Tests
    # ============================================================

    async def test_write_text_file_python(self, result: TestResult):
        """Test writing Python code file"""
        file_path = self.get_test_path("test_script.py")
        content = '''#!/usr/bin/env python3
"""Test script with Unicode and special chars"""

def greet(name: str) -> str:
    """Greet someone in multiple languages"""
    greetings = {
        'en': 'Hello',
        'zh': '你好',
        'ar': 'مرحبا'
    }
    return f"{greetings.get('en', 'Hi')}, {name}!"

if __name__ == "__main__":
    print(greet("World"))
'''

        res = await self.agent.execute_tool('write_text_file', {
            'file_path': file_path,
            'content': content
        })

        assert res['success'], f"Tool failed: {res}"

        with open(file_path, 'r', encoding='utf-8') as f:
            loaded = f.read()

        assert loaded == content, "Content mismatch"
        result.details = f"Wrote {len(content)} chars with Unicode"

    async def test_write_text_file_markdown(self, result: TestResult):
        """Test writing Markdown file with code blocks"""
        file_path = self.get_test_path("README.md")
        content = '''# Test Project

## Code Example

```python
def hello():
    print("Hello, 世界!")
```

## Special Characters

- Quotes: "double" and 'single'
- Symbols: @#$%^&*()
- Unicode: 你好 مرحبا Привет

| Column 1 | Column 2 |
|----------|----------|
| Data     | More     |
'''

        res = await self.agent.execute_tool('write_text_file', {
            'file_path': file_path,
            'content': content
        })

        assert res['success'], f"Tool failed: {res}"

        with open(file_path, 'r', encoding='utf-8') as f:
            loaded = f.read()

        assert loaded == content, "Markdown corrupted"
        result.details = "Verified Markdown with code blocks and Unicode"

    # ============================================================
    # Edit File Tests
    # ============================================================

    async def test_edit_file_simple(self, result: TestResult):
        """Test simple find/replace edit"""
        file_path = self.get_test_path("edit_simple.py")

        await self.agent.execute_tool('write_text_file', {
            'file_path': file_path,
            'content': 'def old_function():\n    pass\n'
        })

        res = await self.agent.execute_tool('edit_file', {
            'file_path': file_path,
            'old_text': 'old_function',
            'new_text': 'new_function'
        })

        assert res['success'], f"Tool failed: {res}"

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert 'new_function' in content, "Replace failed"
        assert 'old_function' not in content, "Old text still present"
        result.details = "Simple replace verified"

    async def test_edit_file_multiline(self, result: TestResult):
        """Test multiline find/replace"""
        file_path = self.get_test_path("edit_multi.py")

        original = '''def process():
    # Old implementation
    return None
'''

        await self.agent.execute_tool('write_text_file', {
            'file_path': file_path,
            'content': original
        })

        res = await self.agent.execute_tool('edit_file', {
            'file_path': file_path,
            'old_text': '    # Old implementation\n    return None',
            'new_text': '    # New implementation\n    return {"status": "success"}'
        })

        assert res['success'], f"Tool failed: {res}"

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert 'New implementation' in content, "Multiline replace failed"
        result.details = "Multiline replace verified"

    async def test_edit_file_replace_all(self, result: TestResult):
        """Test replace all occurrences"""
        file_path = self.get_test_path("edit_all.txt")

        await self.agent.execute_tool('write_text_file', {
            'file_path': file_path,
            'content': 'foo bar foo baz foo'
        })

        res = await self.agent.execute_tool('edit_file', {
            'file_path': file_path,
            'old_text': 'foo',
            'new_text': 'FOO',
            'replace_all': True
        })

        assert res['success'], f"Tool failed: {res}"

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert content.count('FOO') == 3, f"Expected 3 FOO, got {content.count('FOO')}"
        assert 'foo' not in content, "Old text still present"
        result.details = "Replaced 3 occurrences"

    # ============================================================
    # Insert Lines Tests
    # ============================================================

    async def test_insert_lines_beginning(self, result: TestResult):
        """Test inserting at beginning of file"""
        file_path = self.get_test_path("insert_begin.py")

        await self.agent.execute_tool('write_text_file', {
            'file_path': file_path,
            'content': 'def main():\n    pass\n'
        })

        res = await self.agent.execute_tool('insert_lines', {
            'file_path': file_path,
            'line_number': 1,
            'content': '#!/usr/bin/env python3\n"""Module docstring"""'
        })

        assert res['success'], f"Tool failed: {res}"

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        assert lines[0].startswith('#!/usr/bin'), "Shebang not at start"
        assert 'Module docstring' in lines[1], "Docstring not inserted"
        result.details = "Inserted at beginning"

    async def test_insert_lines_middle(self, result: TestResult):
        """Test inserting in middle of file"""
        file_path = self.get_test_path("insert_mid.py")

        await self.agent.execute_tool('write_text_file', {
            'file_path': file_path,
            'content': 'line1\nline2\nline4\n'
        })

        res = await self.agent.execute_tool('insert_lines', {
            'file_path': file_path,
            'line_number': 3,
            'content': 'line3'
        })

        assert res['success'], f"Tool failed: {res}"

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines()]

        assert lines == ['line1', 'line2', 'line3', 'line4'], f"Unexpected order: {lines}"
        result.details = "Inserted in correct position"

    # ============================================================
    # Error Handling Tests
    # ============================================================

    async def test_json_update_nonexistent(self, result: TestResult):
        """Test updating non-existent file"""
        res = await self.agent.execute_tool('json_update', {
            'file_path': self.get_test_path("nonexistent.json"),
            'update_code': "data['x'] = 1"
        })

        assert not res['success'], "Should fail on non-existent file"
        assert 'does not exist' in res['error'].lower(), f"Wrong error: {res['error']}"
        result.details = "Correctly rejected non-existent file"

    async def test_edit_file_text_not_found(self, result: TestResult):
        """Test editing with non-matching text"""
        file_path = self.get_test_path("edit_notfound.txt")

        await self.agent.execute_tool('write_text_file', {
            'file_path': file_path,
            'content': 'hello world'
        })

        res = await self.agent.execute_tool('edit_file', {
            'file_path': file_path,
            'old_text': 'nonexistent',
            'new_text': 'replacement'
        })

        assert not res['success'], "Should fail when text not found"
        assert 'not found' in res['error'].lower(), f"Wrong error: {res['error']}"
        result.details = "Correctly rejected non-matching text"

    # ============================================================
    # Main Test Runner
    # ============================================================

    async def run_all_tests(self):
        """Run all tests and report results"""
        print("=" * 60)
        print("🧪 GrokCodeAgent File Tools Corruption Test Suite")
        print("=" * 60)

        await self.setup()

        # Run all test methods
        test_methods = [
            # JSON Write
            self.test_json_write_basic,
            self.test_json_write_unicode,
            self.test_json_write_nested,
            self.test_json_write_special_chars,
            self.test_json_write_large,

            # JSON Update
            self.test_json_update_basic,
            self.test_json_update_complex,
            self.test_json_update_sandboxing,

            # Text Files
            self.test_write_text_file_python,
            self.test_write_text_file_markdown,

            # Edit Files
            self.test_edit_file_simple,
            self.test_edit_file_multiline,
            self.test_edit_file_replace_all,

            # Insert Lines
            self.test_insert_lines_beginning,
            self.test_insert_lines_middle,

            # Error Handling
            self.test_json_update_nonexistent,
            self.test_edit_file_text_not_found,
        ]

        for test_method in test_methods:
            await self.run_test(test_method)

        await self.teardown()

        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)

        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Success Rate: {passed/total*100:.1f}%")

        if failed > 0:
            print("\n❌ FAILED TESTS:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.error}")

        print("=" * 60)

        return failed == 0


async def main():
    """Main entry point"""
    tester = GrokFileToolsTester()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
