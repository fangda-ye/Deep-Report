# DeepDiver_pipeline/llms/reasoning_openai.py
"""
Simplified OpenAI model wrapper with reasoning support.
Based on actual testing:
- Qwen series: thinking mode requires stream=True, otherwise errors.
- DeepSeek: thinking and stream are independent; R1 natively supports thinking, V3 does not.
"""
import logging
import httpx
import json
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from openai import OpenAI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ReasoningOpenAI(BaseChatModel):
    """
    OpenAI model wrapper with reasoning support.
    Handles Qwen calling modes:
    1. stream + thinking -> normal, with CoT, streaming output
    2. stream + nonthinking -> normal, no CoT, streaming output
    3. nonstream + thinking -> error -> internally converts to stream
    4. nonstream + nonthinking -> normal, direct output
    """
    
    def __init__(
        self,
        model: str = "deepseek-r1",
        api_key: str = "",
        base_url: str = "",
        http_proxy: Optional[str] = None,
        max_tokens: int = 8192,
        timeout: int = 60000,
        temperature: Optional[float] = None,
        enable_reasoning: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Use object.__setattr__ to bypass Pydantic field restrictions
        object.__setattr__(self, 'model_name', model)
        object.__setattr__(self, 'api_key_val', api_key)
        object.__setattr__(self, 'base_url_val', base_url)
        object.__setattr__(self, 'http_proxy_val', http_proxy)
        object.__setattr__(self, 'max_tokens_val', max_tokens)
        object.__setattr__(self, 'timeout_val', timeout)
        object.__setattr__(self, 'temperature_val', temperature)
        object.__setattr__(self, 'enable_reasoning', enable_reasoning)
        
        # Determine strategy based on model type
        object.__setattr__(self, 'is_qwen_model', 'qwen' in model.lower())
        object.__setattr__(self, 'is_deepseek_r1', 'r1' in model.lower())
        object.__setattr__(self, 'supports_reasoning', self.is_qwen_model or self.is_deepseek_r1)
        
        # Determine if using Dashscope API or local vLLM deployment based on base_url
        is_dashscope_api = bool(base_url and ('dashscope' in base_url.lower() or 'aliyun' in base_url.lower()))
        is_local_vllm = bool(base_url and ('localhost' in base_url.lower() or '127.0.0.1' in base_url or '0.0.0.0' in base_url or ':8000' in base_url))
        
        object.__setattr__(self, 'is_dashscope_api', is_dashscope_api)
        object.__setattr__(self, 'is_local_vllm', is_local_vllm)
        
        # Set up HTTP client
        http_client = None
        if http_proxy:
            transport = httpx.HTTPTransport(proxy=http_proxy)
            http_client = httpx.Client(transport=transport, timeout=float(timeout))
        
        # Create OpenAI client
        openai_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client
        )
        object.__setattr__(self, 'client', openai_client)
        
    
    def _parse_think_tags(self, content: str) -> Tuple[str, str]:
        """Parse <think>...</think> tags from local vLLM responses.
        Returns: (reasoning_content, final_content)"""
        if not content or '<think>' not in content:
            return "", content
        
        import re
        
        # Match <think>...</think> tags (supports multiline)
        think_pattern = r'<think>(.*?)</think>'
        matches = re.findall(think_pattern, content, re.DOTALL)
        
        if matches:
            reasoning_content = '\n'.join(matches).strip()
            final_content = re.sub(think_pattern, '', content, flags=re.DOTALL).strip()

            logger.debug(f"Parsed think tags: reasoning={len(reasoning_content)} chars, final={len(final_content)} chars")
            return reasoning_content, final_content
        
        return "", content
    
    def _get_qwen_extra_body(self, enable_thinking: bool) -> Dict[str, Any]:
        """Return the appropriate extra_body config based on API type."""
        if self.is_dashscope_api:
            # Dashscope API format
            return {"enable_thinking": enable_thinking}
        elif self.is_local_vllm:
            # Local vLLM format
            return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        else:
            # Default to Dashscope format
            return {"enable_thinking": enable_thinking}
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response, handling various model/config combinations."""
        openai_messages = []
        for msg in messages:
            msg_dict = self._convert_message_to_dict(msg)
            openai_messages.append(msg_dict)
        
        try:
            if self.is_qwen_model:
                return self._generate_qwen(openai_messages, stop, **kwargs)
            elif self.is_deepseek_r1:
                return self._generate_deepseek_r1(openai_messages, stop, **kwargs)
            else:
                return self._generate_standard(openai_messages, stop, **kwargs)
                
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            raise RuntimeError(f"OpenAI API call failed: {e}")
    
    def _generate_qwen(
        self,
        openai_messages: List[Dict],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResult:
        """Qwen-specific generation logic handling four calling modes."""
        base_params = {
            "model": self.model_name,
            "messages": openai_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens_val),
            "timeout": self.timeout_val,
        }
        
        if self.temperature_val is not None:
            base_params["temperature"] = self.temperature_val
        
        if stop:
            base_params["stop"] = stop
        
        if self.enable_reasoning:
            return self._qwen_stream_collect(base_params, enable_thinking=True)
        else:
            return self._qwen_standard_call(base_params)
    
    def _qwen_stream_collect(self, base_params: Dict, enable_thinking: bool) -> ChatResult:
        """Qwen streaming call that collects results."""
        stream_params = {
            **base_params,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": self._get_qwen_extra_body(enable_thinking)
        }
        
        if hasattr(self, 'bound_tools') and self.bound_tools:
            stream_params["tools"] = self.bound_tools
            stream_params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**stream_params)
        reasoning_content = ""
        content = ""
        finish_reason = None
        usage_data = None
        tool_calls_dict = {}
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                
                # Collect reasoning content
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content
                
                # Collect regular content
                if delta.content is not None:
                    content += delta.content
                
                # Collect tool calls (streaming, need to merge chunks)
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    logger.debug(f"Found {len(delta.tool_calls)} tool call chunks in delta")
                    for tool_call_chunk in delta.tool_calls:
                        # Get tool call index and ID
                        chunk_index = getattr(tool_call_chunk, 'index', None)
                        chunk_id = getattr(tool_call_chunk, 'id', None)
                        
                        # If chunk ID is empty, use index as key and try to find existing tool call
                        if not chunk_id and chunk_index is not None:
                            # Try to find existing tool call with same index
                            existing_tool_id = None
                            for existing_id, existing_data in tool_calls_dict.items():
                                if existing_data.get('index') == chunk_index:
                                    existing_tool_id = existing_id
                                    break
                            
                            if existing_tool_id:
                                tool_id = existing_tool_id
                            else:
                                tool_id = f'call_{chunk_index}'
                        elif chunk_id:
                            tool_id = chunk_id
                        else:
                            tool_id = f'call_{len(tool_calls_dict)}'
                        
                        logger.debug(f"Processing tool call chunk: original_id={repr(chunk_id)}, index={chunk_index}, using_id={tool_id}")
                        
                        # Create tool call entry if it doesn't exist yet
                        if tool_id not in tool_calls_dict:
                            tool_calls_dict[tool_id] = {
                                'id': tool_id,
                                'index': chunk_index,
                                'type': getattr(tool_call_chunk, 'type', 'function'),
                                'function': {
                                    'name': None,
                                    'arguments': ''
                                }
                            }
                        
                        # Merge function info
                        if hasattr(tool_call_chunk, 'function') and tool_call_chunk.function:
                            func = tool_call_chunk.function
                            # Function name usually appears in the first chunk
                            if hasattr(func, 'name') and func.name:
                                tool_calls_dict[tool_id]['function']['name'] = func.name
                                logger.debug(f"Set tool name: {func.name}")
                            # Arguments may be chunked, need to accumulate
                            if hasattr(func, 'arguments') and func.arguments is not None:
                                tool_calls_dict[tool_id]['function']['arguments'] += func.arguments
                                logger.debug(f"Appended arguments: {repr(func.arguments)}")
                
                # Get finish reason
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
            
            # Get usage info (typically in the last chunk)
            if hasattr(chunk, 'usage') and chunk.usage:
                usage_data = chunk.usage
        
        # Convert merged tool calls to list format
        tool_calls = []
        for tool_id, tool_call_data in tool_calls_dict.items():
            # Create an OpenAI-response-like object structure
            class ToolCallObj:
                def __init__(self, data):
                    self.id = data['id']
                    self.type = data['type']
                    self.function = type('Function', (), {
                        'name': data['function']['name'],
                        'arguments': data['function']['arguments']
                    })()
            
            tool_calls.append(ToolCallObj(tool_call_data))
        
        # Handle local vLLM <think> tag format
        if self.is_local_vllm and content and not reasoning_content:
            parsed_reasoning, final_content = self._parse_think_tags(content)
            if parsed_reasoning:
                reasoning_content = parsed_reasoning
                content = final_content
        if tool_calls:
            logger.debug(f"Tool calls collected: {tool_calls}")
        
        return self._build_chat_result(content, reasoning_content, finish_reason, usage_data, tool_calls)
    
    def _qwen_standard_call(self, base_params: Dict) -> ChatResult:
        """Qwen standard call (no reasoning needed)."""
        standard_params = {
            **base_params,
            "extra_body": self._get_qwen_extra_body(False)
        }
        
        if hasattr(self, 'bound_tools') and self.bound_tools:
            standard_params["tools"] = self.bound_tools
            standard_params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**standard_params)
        choice = response.choices[0]
        content = choice.message.content or ""
        reasoning_content = ""
        tool_calls = getattr(choice.message, 'tool_calls', None)
        
        # Fix tool_calls format: convert string args to dict
        if tool_calls:
            for tool_call in tool_calls:
                if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments'):
                    args_raw = tool_call.function.arguments
                    if isinstance(args_raw, str):
                        try:
                            import json
                            parsed_args = json.loads(args_raw)
                            object.__setattr__(tool_call.function, 'arguments', parsed_args)
                        except (json.JSONDecodeError, AttributeError):
                            pass

        # Handle local vLLM <think> tag format
        if self.is_local_vllm and content and not reasoning_content:
            parsed_reasoning, final_content = self._parse_think_tags(content)
            if parsed_reasoning:
                reasoning_content = parsed_reasoning
                content = final_content
        
        return self._build_chat_result(content, reasoning_content, choice.finish_reason, response.usage, tool_calls)
    
    def _generate_deepseek_r1(
        self,
        openai_messages: List[Dict],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResult:
        """DeepSeek R1 generation (natively supports reasoning)."""
        
        request_params = {
            "model": self.model_name,
            "messages": openai_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens_val),
            "stream": False,
            "timeout": self.timeout_val
            # DeepSeek R1 natively supports reasoning, no extra_body needed
        }
        
        if self.temperature_val is not None:
            request_params["temperature"] = self.temperature_val
        
        if stop:
            request_params["stop"] = stop
        
        if hasattr(self, 'bound_tools') and self.bound_tools:
            request_params["tools"] = self.bound_tools
            request_params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**request_params)

        choice = response.choices[0]
        content = choice.message.content or ""
        reasoning_content = getattr(choice.message, 'reasoning_content', '') or ""
        
        tool_calls = getattr(choice.message, 'tool_calls', None)

        if tool_calls:
            for tool_call in tool_calls:
                if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments'):
                    args_raw = tool_call.function.arguments
                    if isinstance(args_raw, str):
                        try:
                            import json
                            parsed_args = json.loads(args_raw)
                            object.__setattr__(tool_call.function, 'arguments', parsed_args)
                        except (json.JSONDecodeError, AttributeError):
                            pass
        
        return self._build_chat_result(content, reasoning_content, choice.finish_reason, response.usage, tool_calls)
    
    def _generate_standard(
        self,
        openai_messages: List[Dict],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResult:
        """Standard call for other models."""
        
        request_params = {
            "model": self.model_name,
            "messages": openai_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens_val),
            "stream": False,
            "timeout": self.timeout_val
        }
        
        if self.temperature_val is not None:
            request_params["temperature"] = self.temperature_val
        
        if stop:
            request_params["stop"] = stop
        
        if hasattr(self, 'bound_tools') and self.bound_tools:
            request_params["tools"] = self.bound_tools
            request_params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**request_params)

        choice = response.choices[0]
        content = choice.message.content or ""
        reasoning_content = ""

        tool_calls = getattr(choice.message, 'tool_calls', None)

        if tool_calls:
            for tool_call in tool_calls:
                if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments'):
                    args_raw = tool_call.function.arguments
                    if isinstance(args_raw, str):
                        try:
                            import json
                            parsed_args = json.loads(args_raw)
                            object.__setattr__(tool_call.function, 'arguments', parsed_args)
                        except (json.JSONDecodeError, AttributeError):
                            pass
        
        return self._build_chat_result(content, reasoning_content, choice.finish_reason, response.usage, tool_calls)
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        """Stream-generate responses with true streaming output."""
        openai_messages = []
        for msg in messages:
            msg_dict = self._convert_message_to_dict(msg)
            openai_messages.append(msg_dict)
        
        try:
            if self.is_qwen_model:
                yield from self._stream_qwen(openai_messages, stop, **kwargs)
            elif self.is_deepseek_r1:
                yield from self._stream_deepseek_r1(openai_messages, stop, **kwargs)
            else:
                yield from self._stream_standard(openai_messages, stop, **kwargs)
                
        except Exception as e:
            logger.error(f"Streaming call failed: {e}")
            raise RuntimeError(f"Streaming call failed: {e}")
    
    def _stream_qwen(
        self,
        openai_messages: List[Dict],
        stop: Optional[List[str]] = None,
        **kwargs
    ):
        """Streaming output for Qwen models."""
        base_params = {
            "model": self.model_name,
            "messages": openai_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens_val),
            "stream": True,
            "stream_options": {"include_usage": True},
            "timeout": self.timeout_val,
        }
        
        if self.temperature_val is not None:
            base_params["temperature"] = self.temperature_val
        
        if stop:
            base_params["stop"] = stop
        
        if self.enable_reasoning:
            base_params["extra_body"] = self._get_qwen_extra_body(True)
        else:
            base_params["extra_body"] = self._get_qwen_extra_body(False)
        if hasattr(self, 'bound_tools') and self.bound_tools:
            base_params["tools"] = self.bound_tools
            base_params["tool_choice"] = "auto"
        
        response = self.client.chat.completions.create(**base_params)
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                
                # Build streaming chunk content
                chunk_content = ""
                additional_kwargs = {}
                tool_call_chunks = []
                
                # Process reasoning content
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    additional_kwargs['reasoning_content'] = delta.reasoning_content
                
                if delta.content is not None:
                    chunk_content = delta.content

                    # For local vLLM, may need to extract <think> tags from content
                    if self.is_local_vllm and '<think>' in chunk_content:
                        # Simple marker; full processing done on complete response
                        additional_kwargs['contains_thinking'] = True
                
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    tool_call_chunks = delta.tool_calls
                if chunk_content or additional_kwargs or tool_call_chunks:
                    from langchain_core.messages import AIMessageChunk
                    from langchain_core.outputs import ChatGenerationChunk
                    
                    ai_chunk = AIMessageChunk(
                        content=chunk_content,
                        additional_kwargs=additional_kwargs,
                        tool_call_chunks=tool_call_chunks
                    )
                    
                    yield ChatGenerationChunk(message=ai_chunk)
    
    def _stream_deepseek_r1(
        self,
        openai_messages: List[Dict],
        stop: Optional[List[str]] = None,
        **kwargs
    ):
        """Streaming output for DeepSeek R1."""
        
        request_params = {
            "model": self.model_name,
            "messages": openai_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens_val),
            "stream": True,
            "timeout": self.timeout_val,
            # DeepSeek R1 natively supports reasoning, no extra_body needed
        }
        
        if self.temperature_val is not None:
            request_params["temperature"] = self.temperature_val
        
        if stop:
            request_params["stop"] = stop
        
        if hasattr(self, 'bound_tools') and self.bound_tools:
            request_params["tools"] = self.bound_tools
            request_params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**request_params)

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                
                chunk_content = ""
                additional_kwargs = {}
                tool_call_chunks = []
                
                # DeepSeek R1 reasoning content
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    additional_kwargs['reasoning_content'] = delta.reasoning_content
                
                # Regular content
                if delta.content is not None:
                    chunk_content = delta.content
                
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    tool_call_chunks = delta.tool_calls

                if chunk_content or additional_kwargs or tool_call_chunks:
                    from langchain_core.messages import AIMessageChunk
                    from langchain_core.outputs import ChatGenerationChunk

                    ai_chunk = AIMessageChunk(
                        content=chunk_content,
                        additional_kwargs=additional_kwargs,
                        tool_call_chunks=tool_call_chunks
                    )

                    yield ChatGenerationChunk(message=ai_chunk)

    def _stream_standard(
        self,
        openai_messages: List[Dict],
        stop: Optional[List[str]] = None,
        **kwargs
    ):
        """Standard streaming output for other models."""
        
        request_params = {
            "model": self.model_name,
            "messages": openai_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens_val),
            "stream": True,
            "timeout": self.timeout_val,
        }
        
        if self.temperature_val is not None:
            request_params["temperature"] = self.temperature_val
        
        if stop:
            request_params["stop"] = stop
        
        if hasattr(self, 'bound_tools') and self.bound_tools:
            request_params["tools"] = self.bound_tools
            request_params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**request_params)

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                
                chunk_content = ""
                tool_call_chunks = []
                
                if delta.content is not None:
                    chunk_content = delta.content

                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    tool_call_chunks = delta.tool_calls
                
                if chunk_content or tool_call_chunks:
                    from langchain_core.messages import AIMessageChunk
                    from langchain_core.outputs import ChatGenerationChunk
                    
                    ai_chunk = AIMessageChunk(
                        content=chunk_content,
                        tool_call_chunks=tool_call_chunks
                    )
                    yield ChatGenerationChunk(message=ai_chunk)
    
    def _build_chat_result(
        self, 
        content: str, 
        reasoning_content: str, 
        finish_reason: Optional[str],
        usage_data: Optional[Any],
        tool_calls: Optional[List[Dict]] = None
    ) -> ChatResult:
        """Build a ChatResult object."""
        usage_metadata = None
        if usage_data:
            usage_metadata = UsageMetadata(
                input_tokens=getattr(usage_data, 'prompt_tokens', 0),
                output_tokens=getattr(usage_data, 'completion_tokens', 0),
                total_tokens=getattr(usage_data, 'total_tokens', 0),
            )
        
        # Build AIMessage
        additional_kwargs = {}
        if reasoning_content:
            additional_kwargs['reasoning_content'] = reasoning_content
        
        # Process tool calls
        formatted_tool_calls = []
        if tool_calls:
            for i, tool_call in enumerate(tool_calls):
                try:
                    # Debug: print tool call structure
                    logger.debug(f"Processing tool call {i}: id={getattr(tool_call, 'id', 'N/A')}, "
                               f"type={getattr(tool_call, 'type', 'N/A')}")
                    
                    # Check tool call object structure
                    if not hasattr(tool_call, 'function'):
                        logger.warning(f"Tool call {i} missing 'function' attribute")
                        continue
                    
                    function = tool_call.function
                    if not hasattr(function, 'name') or not function.name:
                        logger.warning(f"Tool call {i} function missing or empty 'name' attribute")
                        continue
                    
                    if not hasattr(function, 'arguments'):
                        logger.warning(f"Tool call {i} function missing 'arguments' attribute")
                        continue
                    
                    # Parse arguments
                    args_raw = function.arguments
                    logger.debug(f"Tool call {i} ({function.name}) raw arguments: {repr(args_raw)}")
                    
                    if isinstance(args_raw, str):
                        args_raw = args_raw.strip()
                        if not args_raw:
                            args = {}
                        else:
                            try:
                                import json
                                args = json.loads(args_raw)
                            except json.JSONDecodeError as json_err:
                                args = {"query": args_raw}
                    else:
                        args = args_raw if args_raw is not None else {}
                    
                    # Get tool call ID
                    tool_id = getattr(tool_call, 'id', f'call_{i}')
                    
                    formatted_tool_call = {
                        "name": function.name,
                        "args": args,
                        "id": tool_id
                    }
                    formatted_tool_calls.append(formatted_tool_call)
                    logger.debug(f"Successfully processed tool call {i}: {formatted_tool_call['name']} with {len(str(args))} chars args")
                    
                except Exception as e:
                    logger.warning(f"Failed to parse tool call {i}: {e}")
        
        ai_message = AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            response_metadata={
                "model_name": self.model_name,
                "finish_reason": finish_reason,
            },
            usage_metadata=usage_metadata,
            tool_calls=formatted_tool_calls or []
        )
        
        return ChatResult(generations=[ChatGeneration(message=ai_message)])
    
    @property
    def _llm_type(self) -> str:
        return "reasoning_openai"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "base_url": self.base_url_val,
            "max_tokens": self.max_tokens_val,
            "temperature": self.temperature_val,
            "supports_reasoning": self.supports_reasoning
        }
    
    @property
    def _llm_type(self) -> str:
        return "reasoning_openai"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "base_url": self.base_url_val,
            "max_tokens": self.max_tokens_val,
            "temperature": self.temperature_val,
            "supports_reasoning": self.supports_reasoning
        }
    
    def bind_tools(self, tools: List[Any]) -> "ReasoningOpenAI":
        """Bind tools to the model."""
        openai_tools = []
        for tool in tools:
            tool_schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    # Simplified parameter structure
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The input to the tool"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
            openai_tools.append(tool_schema)
        
        # Create new instance with tool configuration
        new_instance = ReasoningOpenAI(
            model=self.model_name,
            api_key=self.api_key_val,
            base_url=self.base_url_val,
            http_proxy=self.http_proxy_val,
            max_tokens=self.max_tokens_val,
            timeout=self.timeout_val,
            temperature=self.temperature_val,
            enable_reasoning=self.enable_reasoning
        )
        
        # Store tool info
        object.__setattr__(new_instance, 'bound_tools', openai_tools)
        object.__setattr__(new_instance, 'tools_map', {tool.name: tool for tool in tools})
        
        return new_instance
    
    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        """Convert LangChain message to OpenAI format."""
        message_dict = {"content": message.content}
        
        if isinstance(message, HumanMessage):
            message_dict["role"] = "user"
        elif isinstance(message, SystemMessage):
            message_dict["role"] = "system"
        elif isinstance(message, AIMessage):
            message_dict["role"] = "assistant"
            
            # Handle tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls_list = []
                for tc in message.tool_calls:
                    tool_call_dict = {
                        "id": tc.get("id", f"call_{len(tool_calls_list)}"),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["args"] if isinstance(tc["args"], str) else json.dumps(tc["args"])
                        }
                    }
                    tool_calls_list.append(tool_call_dict)
                message_dict["tool_calls"] = tool_calls_list
            
            # If there are tool calls, content may need to be None
            if "tool_calls" in message_dict:
                message_dict["content"] = message_dict["content"] or None
                
        elif isinstance(message, ToolMessage):
            message_dict["role"] = "tool"
            message_dict["tool_call_id"] = getattr(message, 'tool_call_id', 'unknown')
        else:
            message_dict["role"] = "user"
            
        return message_dict


def create_reasoning_openai_llm(
    model: str,
    api_key: str,
    base_url: str,
    http_proxy: Optional[str] = None,
    max_tokens: int = 8192,
    timeout: int = 600,
    temperature: Optional[float] = None,
    enable_reasoning: bool = True
) -> ReasoningOpenAI:
    """Create a reasoning-capable OpenAI LLM."""
    
    return ReasoningOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        http_proxy=http_proxy,
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
        enable_reasoning=enable_reasoning
    )

