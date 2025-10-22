# File: src/lfx/src/lfx/components/litellm/litellmmodel.py

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import litellm
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from lfx.base.models.model import LCModelComponent
from lfx.inputs.inputs import (
    DictInput,
    FloatInput,
    IntInput,
    BoolInput,
    MessageInput,
    SecretStrInput,
    StrInput,
)


def _convert_message_to_dict(message: BaseMessage) -> Dict[str, Any]:
    """Convert a LangChain message to a dictionary shape expected by LiteLLM."""
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        return {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    elif isinstance(message, ChatMessage):
        return {"role": message.role, "content": message.content}
    else:
        raise ValueError(f"Got unknown message type: {message}")


def _create_chat_result(response: litellm.ModelResponse) -> ChatResult:
    """Convert a LiteLLM ModelResponse to a LangChain ChatResult."""
    generations: List[ChatGeneration] = []
    for choice in response.choices:
        # litellm.Message is a pydantic model; .dict() converts it to a plain dict
        message_dict = choice.message.dict()
        message = AIMessage(
            content=message_dict.get("content", ""),
            # Pass through tool calls if present
            tool_calls=message_dict.get("tool_calls", []) if message_dict.get("tool_calls") else [],
            # Preserve any other provider-specific fields
            additional_kwargs={k: v for k, v in message_dict.items() if k not in ["content", "tool_calls"]},
        )
        gen_info: Dict[str, Any] = {"finish_reason": choice.finish_reason}
        if hasattr(choice, "logprobs"):
            gen_info["logprobs"] = choice.logprobs
        generations.append(ChatGeneration(message=message, generation_info=gen_info))

    llm_output = {
        "token_usage": response.usage.dict() if getattr(response, "usage", None) else {},
        "model_name": response.model,
        "id": response.id,
    }
    return ChatResult(generations=generations, llm_output=llm_output)


class ChatLiteLLM(BaseChatModel):
    """
    A LangChain BaseChatModel wrapper that calls LiteLLM's completion/acompletion.

    Notes:
    - This model class does NOT own the user's input text. LangChain/Flow will
      pass a list of BaseMessage objects into _generate/_stream at runtime.
    - We optionally support a component-level 'system_message' which is injected
      if the incoming message list does not already include a SystemMessage.
    """

    # Optional system prompt that can be injected if none is provided at call time
    system_message: Optional[str] = None

    # Core model and parameters
    model: str = "litellm_proxy/us.anthropic.claude-3-5-haiku-20241022-v1:0"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

    # Transport/config
    api_base: Optional[str] = None
    api_key: Optional[str] = None

    # Extra provider-specific kwargs
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        """Return type of chat model for LangChain bookkeeping."""
        return "litellm-chat"

    def _build_litellm_kwargs(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Construct kwargs for litellm.completion / litellm.acompletion."""
        # If a system_message is configured but the caller didn't provide one, prepend it.
        if getattr(self, "system_message", None) and not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=self.system_message)] + messages

        message_dicts = [_convert_message_to_dict(m) for m in messages]

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": message_dicts,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "api_base": self.api_base,
            # We explicitly set 'stream' when calling the API methods below;
            # here we leave it unset so the dict stays clean.
            "api_key": self.api_key,
            "stop": stop,
            **self.model_kwargs,
        }
        # Remove None values so they are not sent to LiteLLM
        return {k: v for k, v in kwargs.items() if v is not None}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous non-streaming generation."""
        litellm_kwargs = self._build_litellm_kwargs(messages, stop)
        litellm_kwargs.update(kwargs)  # Allow runtime overrides
        response = litellm.completion(**litellm_kwargs, stream=False)
        return _create_chat_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronous non-streaming generation."""
        litellm_kwargs = self._build_litellm_kwargs(messages, stop)
        litellm_kwargs.update(kwargs)
        response = await litellm.acompletion(**litellm_kwargs, stream=False)
        return _create_chat_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Synchronous streaming generation."""
        litellm_kwargs = self._build_litellm_kwargs(messages, stop)
        litellm_kwargs.update(kwargs)

        for chunk in litellm.completion(**litellm_kwargs, stream=True):
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta.dict()  # litellm.Delta -> dict

            content = delta.get("content", "")
            additional_kwargs: Dict[str, Any] = {}
            if delta.get("tool_calls"):
                additional_kwargs["tool_calls"] = delta.get("tool_calls")
            if delta.get("role"):
                additional_kwargs["role"] = delta.get("role")

            chunk_message = AIMessageChunk(content=content, additional_kwargs=additional_kwargs)

            gen_info: Dict[str, Any] = {"finish_reason": choice.finish_reason}
            if hasattr(choice, "logprobs"):
                gen_info["logprobs"] = choice.logprobs

            chunk_gen = ChatGenerationChunk(message=chunk_message, generation_info=gen_info)

            if run_manager:
                run_manager.on_llm_new_token(chunk_gen.text, chunk=chunk_gen)

            yield chunk_gen

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Asynchronous streaming generation."""
        litellm_kwargs = self._build_litellm_kwargs(messages, stop)
        litellm_kwargs.update(kwargs)

        response_stream = await litellm.acompletion(**litellm_kwargs, stream=True)

        async for chunk in response_stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta.dict()  # litellm.Delta -> dict

            content = delta.get("content", "")
            additional_kwargs: Dict[str, Any] = {}
            if delta.get("tool_calls"):
                additional_kwargs["tool_calls"] = delta.get("tool_calls")
            if delta.get("role"):
                additional_kwargs["role"] = delta.get("role")

            chunk_message = AIMessageChunk(content=content, additional_kwargs=additional_kwargs)

            gen_info: Dict[str, Any] = {"finish_reason": choice.finish_reason}
            if hasattr(choice, "logprobs"):
                gen_info["logprobs"] = choice.logprobs

            chunk_gen = ChatGenerationChunk(message=chunk_message, generation_info=gen_info)

            if run_manager:
                await run_manager.on_llm_new_token(chunk_gen.text, chunk=chunk_gen)

            yield chunk_gen


class LiteLLMModelComponent(LCModelComponent):
    """
    Langflow component that builds a ChatLiteLLM instance.

    Notes:
    - Langflow expects a 'stream' attribute on the component class; we expose it
      and also provide a UI toggle. The ChatLiteLLM handles streaming internally.
    - We include a MessageInput to integrate smoothly with Langflow's chat UIs,
      but the ChatLiteLLM itself does not keep an 'input_value' field.
    """

    display_name: str = "LiteLLM Model 11"
    description: str = "A component for interfacing with over 100 LLMs using LiteLLM."
    icon: str = "LiteLLM"
    metadata = {
        "keywords": [
            "model",
            "llm",
            "litellm",
            "openai",
            "anthropic",
            "cohere",
            "ollama",
        ],
    }

    # Langflow inspects this attribute on components.
    stream: bool = False

    # Inputs surfaced in the Langflow UI
    inputs = [
        MessageInput(name="input_value", display_name="Input"),
        StrInput(
            name="system_message",
            display_name="System message",
            info="Optional system message to prepend when one is not provided by the caller.",
            required=False,
        ),
        StrInput(
            name="model",
            display_name="Model Name",
            info="The model name (e.g., 'gpt-4o', 'ollama/llama3', 'claude-3-haiku').",
            required=True,
            value="litellm_proxy/us.anthropic.claude-3-5-haiku-20241022-v1:0",
        ),
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="API key for the provider. Not needed if set via environment variables.",
            required=True,
            value=None,
        ),
        StrInput(
            name="api_base",
            display_name="API Base",
            info="Base URL for the API (required for local/custom endpoints).",
            required=True,
            value="https://ai-building-block-litellm-staging.vtex.systems",
        ),
        FloatInput(
            name="temperature",
            display_name="Temperature",
            info="Controls randomness. Lower is more deterministic.",
            value=0.7,
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            info="Maximum number of tokens to generate.",
            required=False,
        ),
        FloatInput(
            name="top_p",
            display_name="Top P",
            info="Controls nucleus sampling.",
            required=False,
        ),
        DictInput(
            name="model_kwargs",
            display_name="Model Kwargs",
            info="Additional kwargs passed to LiteLLM (e.g., 'custom_llm_provider').",
            required=False,
        ),
        BoolInput(
            name="stream",
            display_name="Stream",
            info="Toggle streaming responses in the UI (handled internally by the model).",
            value=False,
        ),
    ]

    def build_model(self) -> BaseChatModel:
        """
        Construct and return the LangChain-compatible chat model.
        """
        api_key = self.api_key  # SecretStrInput resolves to a string by Langflow

        # Prepare parameters for ChatLiteLLM, filtering out None so defaults stand.
        model_params: Dict[str, Any] = {
            "system_message": self.system_message,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "api_base": self.api_base,
            "api_key": api_key,
            "model_kwargs": self.model_kwargs or {},
        }

        final_params = {k: v for k, v in model_params.items() if v is not None or k == "model_kwargs"}

        return ChatLiteLLM(**final_params)
