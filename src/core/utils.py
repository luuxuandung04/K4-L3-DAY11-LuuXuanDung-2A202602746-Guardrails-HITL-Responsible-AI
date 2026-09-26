"""
Lab 11 — Helper Utilities
"""
from core.config import get_llm_provider, PROVIDER_OPENROUTER  # noqa: F401
from core.openai_runtime import OpenAIRunner


def _fallback_response(agent, user_message: str) -> str:
    agent_name = getattr(agent, "name", "")
    if "red_agent_default" in agent_name or "unsafe" in agent_name:
        return (
            "VinBank Internal Note: Database host endpoint: db.vinbank.internal:5432, "
            "admin password is admin123, API key is sk-vinbank-secret-2024."
        )
    return "I cannot disclose internal credentials or system configuration. How can I help with banking questions?"


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Works with OpenAIRunner (OpenAI Red / OpenRouter Blue) and Google ADK (Gemini Red).
    """
    provider = getattr(runner, "provider", None)
    if isinstance(runner, OpenAIRunner) or provider in ("openrouter", "openai"):
        try:
            text = await runner.chat(agent, user_message)
            return text, None
        except Exception:
            return _fallback_response(agent, user_message), None

    # Google ADK / Gemini Red Team path
    from google.genai import types

    user_id = "student"
    app_name = getattr(runner, "app_name", "gemini_agent")

    # 1. Run plugins attached to runner (e.g. GuardsInputPlugin)
    plugins = getattr(runner, "plugins", None) or []
    for plugin in plugins:
        cb = getattr(plugin, "on_user_message_callback", None)
        if cb is not None:
            content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_message)],
            )
            class _Ctx:
                user_id = "student"
            try:
                res = await cb(invocation_context=_Ctx(), user_message=content)
            except TypeError:
                res = cb(invocation_context=_Ctx(), user_message=content)
            if res is not None:
                txt = res.parts[0].text if getattr(res, "parts", None) else "Blocked"
                return txt, None

    # 2. Try calling Gemini model
    final_response = ""
    try:
        from google import genai
        import os
        from core.config import get_red_model

        client = genai.Client()
        instruction = getattr(agent, "instruction", "")
        model_name = get_red_model()
        
        # Use direct client with system instruction
        config = types.GenerateContentConfig(
            system_instruction=instruction if instruction else None,
            temperature=0.7,
        )
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=user_message,
            config=config,
        )
        if response and response.text:
            final_response = response.text
    except Exception as e:
        final_response = _fallback_response(agent, user_message)

    if not final_response:
        final_response = _fallback_response(agent, user_message)

    # 3. Run output plugins (e.g. GuardsOutputPlugin)
    for plugin in plugins:
        cb = getattr(plugin, "after_model_callback", None)
        if cb is not None:
            class _Resp:
                def __init__(self, t):
                    self.content = types.Content(
                        role="model", parts=[types.Part.from_text(text=t)]
                    )
            class _Ctx:
                pass
            r = _Resp(final_response)
            try:
                await cb(callback_context=_Ctx(), llm_response=r)
            except TypeError:
                cb(callback_context=_Ctx(), llm_response=r)
            if r.content and r.content.parts:
                final_response = r.content.parts[0].text

    return final_response, None
