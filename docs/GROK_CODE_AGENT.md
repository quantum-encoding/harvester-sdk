# Grok Code Agent - Agentic Coding Assistant

Powered by **grok-code-fast-1**, the Grok Code Agent is designed for agentic programming workflows with:
- **4x faster** than other agentic models
- **1/10th the cost** of leading alternatives
- **Streaming reasoning traces** - See the model's thinking process
- **Native tool calling** - File ops, code search, command execution
- **Context optimization** - XML/Markdown formatting + cache hits

## Quick Start

```bash
# Simple coding task
harvester grok-code "Add error handling to auth.py"

# With context files
harvester grok-code "Refactor database module" -f db.py -f models.py

# Debugging with reasoning traces
harvester grok-code "Fix memory leak in worker" -t debugging --show-reasoning

# Feature development with project structure
harvester grok-code "Add rate limiting API" -p structure.txt -t feature
```

## Features

### 🧠 Streaming Reasoning Traces
See the model's step-by-step thinking process as it works through your task:

```bash
harvester grok-code "Optimize query performance" --show-reasoning
```

Output:
```
💭 Reasoning Traces:
--- Step 1 ---
First, I need to understand the current query structure.
Let me search for SQL queries in the codebase...

--- Step 2 ---
Found the problematic query in db.py:145. It's missing an index.
I'll need to check the database schema...
```

### 🔧 Native Tool Calling
The agent has access to powerful tools:

- **read_file** - Read file contents
- **write_file** - Create or update files
- **search_code** - Search for code patterns (regex)
- **list_files** - List files matching a pattern
- **execute_command** - Run shell commands

Example agentic workflow:
1. Search for authentication code
2. Read relevant files
3. Identify issues
4. Write improved code
5. Run tests to verify

### 📝 Context-Aware Prompting

Provide rich context using files and project structure:

```bash
harvester grok-code "Add OAuth support" \
  -f auth.py \
  -f config.py \
  -p "Project uses Express.js with PostgreSQL backend" \
  -t feature
```

The agent formats context with XML/Markdown for clarity:

```xml
<file path="auth.py">
\`\`\`python
# authentication code...
\`\`\`
</file>

<project_structure>
Express.js with PostgreSQL backend
</project_structure>
```

### ⚡ Cache Optimization

The agent preserves message history prefix for automatic cache hits:
- **First iteration**: Full context processing
- **Subsequent iterations**: Cached prefix + new tool results
- **Result**: Blazing fast multi-step workflows

## Task Types

Choose the right task type for optimized system prompts:

### General (default)
```bash
harvester grok-code "Add logging to API endpoints"
```

### Debugging
```bash
harvester grok-code "Fix race condition in worker pool" -t debugging
```
- Systematic root cause identification
- Search tools to find related code
- Thorough testing of fixes

### Refactoring
```bash
harvester grok-code "Refactor monolithic service" -t refactoring
```
- Preserve functionality
- Improve code quality
- Follow project conventions
- Update documentation

### Feature Development
```bash
harvester grok-code "Add WebSocket support" -t feature
```
- Design before implementing
- Handle edge cases and errors
- Write clean, testable code
- Document new functionality

## CLI Options

```
Usage: harvester grok-code [OPTIONS] TASK

Options:
  -f, --files PATH               Files to include as context (multiple)
  -p, --project-structure TEXT   Project structure file or description
  -t, --task-type [general|debugging|refactoring|feature]
  -i, --max-iterations INTEGER   Maximum iterations (default: 10)
  --show-reasoning              Display reasoning traces
  -o, --output PATH             Save result to JSON file
  --help                        Show this message and exit
```

## Examples

### Example 1: Simple Task
```bash
harvester grok-code "Add type hints to util.py"
```

### Example 2: Bug Fix with Context
```bash
harvester grok-code "Fix SQL injection vulnerability" \
  -f api/routes.py \
  -f db/queries.py \
  -t debugging
```

### Example 3: Feature with Full Context
```bash
harvester grok-code "Implement user authentication with JWT" \
  -f auth.py \
  -f models.py \
  -f config.py \
  -p structure.txt \
  -t feature \
  --show-reasoning \
  -o auth_implementation.json
```

Output saved to `auth_implementation.json`:
```json
{
  "task": "Implement user authentication with JWT",
  "task_type": "feature",
  "status": "completed",
  "iterations": 5,
  "result": "...",
  "reasoning_traces": [...],
  "tool_calls": [...]
}
```

### Example 4: Large Refactoring
```bash
harvester grok-code "Split monolithic API into microservices" \
  -f app.py \
  -f $(find . -name "*.py" -path "./api/*") \
  -p "Migrate to microservices: auth, payments, notifications" \
  -t refactoring \
  -i 20 \
  --show-reasoning
```

## Best Practices

### ✅ DO:
- **Provide specific context** - Include relevant files and structure
- **Set clear goals** - Detailed requirements get better results
- **Iterate and refine** - Cheap iterations allow experimentation
- **Use appropriate task type** - Gets optimized system prompt
- **Show reasoning for complex tasks** - Understand the approach

### ❌ DON'T:
- **Avoid vague prompts** - "Make it better" won't work well
- **Don't overload context** - Only include relevant files
- **Don't skip project structure** - Helps model understand constraints
- **Don't use for one-shot Q&A** - Use Grok 4 instead for that

## When to Use Grok Code vs Grok 4

### Use **Grok Code Agent** for:
- ✅ Multi-step coding tasks
- ✅ Navigating large codebases
- ✅ Agentic workflows with tools
- ✅ Iterative development
- ✅ Cost-effective experimentation (1/10th cost)

### Use **Grok 4** for:
- ✅ One-shot Q&A
- ✅ Complex concept explanations
- ✅ Deep debugging with all context upfront
- ✅ Structured output generation

## Architecture

### Agent Components:

1. **Context Builder**
   - Formats files with XML/Markdown tags
   - Adds project structure and dependencies
   - Optimizes for model understanding

2. **Streaming Engine**
   - Extracts `reasoning_content` from stream
   - Displays thinking process in real-time
   - Handles tool calls during reasoning

3. **Tool Executor**
   - File operations (read/write)
   - Code search (grep/ripgrep)
   - Command execution
   - Results fed back to model

4. **Cache Optimizer**
   - Preserves message prefix
   - Maximizes cache hits
   - 4x faster iterations

### Workflow:
```
User Task → Context Builder → System Prompt
                                    ↓
                              Initial Request
                                    ↓
                         ← Streaming Response →
                         (Reasoning + Tool Calls)
                                    ↓
                            Execute Tools
                                    ↓
                         Tool Results → Model
                                    ↓
                         More Tool Calls? → Yes (loop)
                                    ↓ No
                              Final Result
```

## Troubleshooting

### Issue: "Max iterations reached"
**Solution**: Increase max iterations or break task into smaller steps
```bash
harvester grok-code "Complex task" -i 20
```

### Issue: Slow iterations
**Solution**: Don't modify message history (breaks cache)
- Agent automatically preserves prefix for cache hits

### Issue: Poor results
**Solution**: Add more context and be specific
```bash
# Bad
harvester grok-code "Fix bug"

# Good
harvester grok-code "Fix race condition in payment processor" \
  -f payment.py -f queue.py -t debugging
```

### Issue: Tool execution fails
**Solution**: Check file paths and permissions
- Paths should be relative to current working directory
- Agent needs read/write permissions for files

## API Usage

Use the agent programmatically:

```python
from agents import GrokCodeAgent
import asyncio

async def main():
    agent = GrokCodeAgent(
        model="grok-code-fast-1",
        max_iterations=10
    )

    # Format context
    context = agent.format_context(
        files={
            'auth.py': open('auth.py').read()
        },
        project_structure="Express.js + PostgreSQL",
        dependencies=['express', 'passport', 'pg']
    )

    # Execute task
    result = await agent.execute_task(
        description="Add OAuth2 authentication",
        context={'files': context},
        task_type='feature'
    )

    print(f"Status: {result.status}")
    print(f"Iterations: {result.iterations}")
    print(f"Result: {result.result}")

    # Access reasoning traces
    for trace in result.reasoning_traces:
        print(f"Step {trace.step_number}: {trace.content}")

asyncio.run(main())
```

## Limitations

- **Max iterations**: 10 by default (configurable)
- **Tool timeout**: Commands timeout after 60s
- **File size**: Large files (>100KB) may impact performance
- **Streaming only**: Reasoning traces require streaming mode
- **No vision**: grok-code-fast-1 is text-only

## Future Enhancements

- [ ] Multi-file editing with diff preview
- [ ] Git integration (commits, branches)
- [ ] Test generation and execution
- [ ] Code review mode
- [ ] Collaborative coding (multiple agents)
- [ ] Visual reasoning trace display

## Learn More

- [xAI Grok Code Docs](https://docs.x.ai/docs/guides/grok-code)
- [Tool/Function Calling](https://docs.x.ai/docs/guides/function-calling)
- [xAI Pricing](https://x.ai/pricing)

---

**Powered by grok-code-fast-1 - 4x faster, 1/10th the cost** 🚀
