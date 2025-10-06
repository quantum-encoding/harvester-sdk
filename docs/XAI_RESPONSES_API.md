# xAI Responses API - Stateful Conversations

xAI offers a **Responses API** for stateful conversations (similar to OpenAI's Responses API).

## Key Features

1. **Stateful** - Server stores conversation history for 30 days
2. **Cost Effective** - Automatic caching reduces billing for long conversations
3. **Encrypted Reasoning** - Can retrieve thinking traces with `include: ["reasoning.encrypted_content"]`
4. **Response Management** - Get/delete stored responses by ID

## Current Status

❌ **NOT IMPLEMENTED** - The harvester SDK currently uses:
- Chat Completions API (`/v1/chat/completions`) for regular requests
- Beta Chat Completions (`/v1/beta/chat/completions`) for structured outputs

## Why Not Implemented?

1. **Requires xAI SDK** - Examples use `xai_sdk.Client` which we don't use
2. **No REST API docs** - Only SDK examples available
3. **Uncertain structured output support** - Docs don't mention if Responses API supports structured outputs
4. **Current solution works** - Chat Completions covers most use cases

## API Differences

| Feature | Chat Completions | Responses API |
|---------|-----------------|---------------|
| State | Stateless | Stateful (30 days) |
| History | Send full context | Server stores |
| Caching | Manual | Automatic |
| Reasoning traces | ❌ | ✅ (encrypted) |
| Continuation | Re-send all messages | Use `previous_response_id` |

## Example Usage (SDK-based)

```python
from xai_sdk import Client

client = Client(api_key=os.getenv("XAI_API_KEY"))

# Create stateful conversation
chat = client.chat.create(model="grok-4", store_messages=True)
chat.append(system("You are a helpful assistant"))
chat.append(user("What is 2+2?"))
response = chat.sample()

# Continue conversation
chat = client.chat.create(
    model="grok-4",
    previous_response_id=response.id,
    store_messages=True
)
chat.append(user("What about 3+3?"))
second_response = chat.sample()

# Retrieve stored response
response = client.chat.get_stored_completion(response.id)

# Delete stored response
client.chat.delete_stored_completion(response.id)
```

## REST API Endpoints (if available)

```
POST /v1/responses          # Create new response
GET  /v1/responses/{id}     # Retrieve response
DELETE /v1/responses/{id}   # Delete response
```

*(Documentation pending - may not be available via REST)*

## Future Implementation

To add Responses API support:

1. Research REST API endpoints (if available)
2. Add `XAIResponsesProvider` class
3. Implement state management (response IDs)
4. Add CLI commands: `continue`, `get-response`, `delete-response`
5. Test structured output compatibility

## References

- [xAI Responses API Docs](https://docs.x.ai/docs/guides/responses)
- Current implementation: `providers/xai_provider.py`
