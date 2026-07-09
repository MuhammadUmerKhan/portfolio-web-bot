# 09 — LLM Gateway (Portkey)

> **One-line summary:** An LLM Gateway is a proxy layer that sits between your application and the LLM provider — adding resilience, observability, and fallback strategies with zero changes to your business logic.

---

## What Is an LLM Gateway?

Without a gateway, your application calls the LLM provider directly (e.g., Groq). If Groq rate-limits you or goes down, your application crashes. 

An LLM Gateway intercepts those calls and acts as a traffic controller:
- **Routes** each request to the correct provider.
- **Retries** automatically on rate limits (429) or server errors (503).
- **Falls back** to a different model or provider if the first one fails entirely.
- **Caches** identical requests to save money and latency.

---

## Our Implementation: Portkey Cloud

We use **Portkey's Managed Cloud Gateway** to wrap our LangChain/LangGraph workflow.

Instead of hardcoding `ChatGroq` deep inside the agent nodes, we use `ChatOpenAI` pointed at Portkey's universal endpoint (`https://api.portkey.ai`). 

### Why ChatOpenAI?
Groq's native LangChain client (`ChatGroq`) is hardwired to `api.groq.com` and cannot be easily proxied. `ChatOpenAI` supports a custom `base_url` and `default_headers`, allowing us to easily inject Portkey's routing instructions without writing custom LangChain classes.

---

## The Fallback Strategy

Our system relies heavily on the open-weights Groq ecosystem for extreme latency reduction. However, Groq's free-tier rate limits can be restrictive. 

To solve this, we use a **Portkey Config ID** (`pc-xxxxxx`) injected via our `.env` file, which tells the gateway to execute the following logic:

1. **Primary Target:** Attempt the request using our primary virtual key (`portfolio-bot`) with the **`llama-3.3-70b-versatile`** model.
2. **Retry:** If Groq returns a `429 Too Many Requests`, Portkey automatically respects the `retry-after` header and attempts the request up to 3 times.
3. **Fallback Target:** If the primary target completely fails, Portkey seamlessly routes the exact same request to our backup virtual key (`portfolio-bot-2`) using the faster but smaller **`llama-3.1-8b-instant`** model.
4. **Caching:** Simple caching is enabled, so identical requests are returned instantly without hitting Groq.

```json
{
  "strategy": {
    "mode": "fallback",
    "on_status_codes": [429, 503]
  },
  "retry": {
    "attempts": 3
  },
  "cache": {
    "mode": "simple"
  },
  "targets": [
    {
      "virtual_key": "portfolio-bot",
      "override_params": {
        "model": "llama-3.3-70b-versatile"
      }
    },
    {
      "virtual_key": "portfolio-bot-2",
      "override_params": {
        "model": "llama-3.1-8b-instant"
      }
    }
  ]
}
```

By decoupling the routing logic into the Portkey Cloud Dashboard, we can hot-swap providers (e.g., adding OpenAI GPT-4o or Anthropic Claude 3.5 Sonnet) at runtime without ever touching or redeploying the application code!
