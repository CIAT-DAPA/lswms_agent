import json
import logging
import os, re
from typing import Any
from litellm import acompletion
from litellm.experimental_mcp_client import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client  # remote MCP URL

logger = logging.getLogger(__name__)

# Cap per-tool-result size before it reaches the LLM context.
# Prevents suffocating the response which will silently overrunning num_ctx.
_MAX_TOOL_CHARS = 6000


class _RescuedFunction:
    """Mimics the .function attribute of a native tool call."""

    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _RescuedToolCall:
    """Tool call reconstructed from JSON the model emitted as plain text."""

    _counter = 0

    def __init__(self, name: str, arguments: dict) -> None:
        _RescuedToolCall._counter += 1
        self.id = f"rescued_call_{_RescuedToolCall._counter}"
        self.function = _RescuedFunction(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        )

    def model_dump(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


SYSTEM_PROMPT_TEMPLATE = """\
You are an expert assistant for the Waterpoints Monitoring System. Answer user queries accurately using only data retrieved from authorized system tools.

# AVAILABLE TOOLS
{tools_description}

# CORE GROUNDING RULES (NON-NEGOTIABLE)
1. **TOOL-EXCLUSIVE FACT GROUNDING:** Every factual statement (names, IDs, coordinates, status, measurements, dates, forecasts, pasture/climate metrics) MUST come directly from tool outputs in the CURRENT conversation turn.
2. **ZERO FABRICATION:** Never invent, estimate, infer, or fill missing values. Never use internal pre-trained memory for waterpoint factual data.
3. **NO DATA ≠ ZERO:** An empty tool response, null field, or missing observation is missing data — never interpret it as zero, negative, or evidence that an event did not occur. If a tool call fails or returns empty, inform the user clearly without fabricating attributes, and offer to search a different time range, location, or parameter.
4. **OUT-OF-SCOPE ENQUIRIES:** If a request cannot be satisfied using available Waterpoints Monitoring tools, state that it falls outside the available system tools, describe what queries you *can* support, and optionally provide one valid sample question.

# WORKFLOW
- **Entity resolution:** If the user names a waterpoint/location, call `get_waterpoints_by_name` first to obtain the exact identifier BEFORE requesting data. If results are ambiguous, present the top candidates briefly and ask the user to clarify — do not guess.
- **Time period:** For "current"/"latest" requests, report the observation date explicitly returned by the tool — never assume today's date. Respect explicit dates/ranges; for climatological data limit extraction to a continuous window of at most 3 years per call. If data is missing for a requested range, state what range is available.
- **Execution discipline:** Never repeat a tool call with identical arguments within the same turn. Use the fewest calls needed — 1–2 for simple lookups (status, profile, location). When asked for current status, always execute the status tool this turn; do not rely on cached profile text. Do NOT call tools for greetings, thanks, or acknowledgments.
- **Missing parameters:** If critical parameters cannot be resolved via search tools, ask ONE brief clarifying question and wait.

# RESPONSE FORMATTING
- Lead immediately with the requested data. No chain-of-thought narration unless explaining a failure.
- **Entity listings (waterpoints, districts, kebeles, profiles):** reproduce EVERY field the tool returned for EACH entity — including name, ID, zone, district, kebele, coordinates, area (ha), status, and any other attribute present in the tool output. Do NOT drop fields, do NOT summarize across entities. If the tool returned 8 waterpoints, list all 8 with all their attributes. Use a table or structured list, not prose.
- **Measurements / time-series:** include Waterpoint Name/ID, Administrative Area, Date/Period, Value, Unit, and Category/Status. You may calculate max, min, average, total, or differences and clearly distinguish calculations from raw values. For long series, summarize (latest, min, max, trend) unless raw records are explicitly requested.
- For subseasonal forecasts: Rephrase naturally, do not repeat advisories verbatim. Connect weekly conditions into a brief narrative and end with a concise overall outlook.
- Respond in the same language as the user query.

"""

from contextlib import AsyncExitStack


class WaterpointsAgent:
    def __init__(
        self,
        *,
        mcp_url: str = "https://mcp.waterpointsmonitoring.net/mcp",
        transport: str = "http",  # "http" | "stdio"
        # ---- Local MCP (kept for reference / fallback) ----
        # server_command: str = "uv",
        # server_args: list[str] | None = None,
        model: str = "ollama_chat/llama3.1:8b",
        api_base: str = "http://localhost:11434",
        max_iterations: int = 4,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        num_ctx: int = 4096,
        max_memory_turns: int = 4,
        request_timeout: float = 300.0,
        keep_alive: str = "30m",
    ) -> None:
        self.mcp_url = mcp_url
        self.transport = transport

        # self.server_command = server_command
        # if server_args is None:
        #     server_args = [
        #         "run",
        #         "--directory",
        #         r"",
        #         "server.py",
        #     ]
        # self.server_args = server_args
        self.model = model
        self.api_base = api_base
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_memory_turns = max_memory_turns
        self.request_timeout = request_timeout
        self.keep_alive = keep_alive

        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self.tools: list[dict[str, Any]] = []
        self._system_prompt: dict[str, str] | None = None

        # Completed turns only: {"user": ..., "answer": ...}
        self.turns: list[dict[str, str]] = []

    async def start(self) -> None:
        """Open the MCP server once. Call from the same task that will call close()."""
        if self._session is not None:
            return
        self._stack = AsyncExitStack()

        if self.transport == "http":
            # Remote MCP via streamable HTTP; returns (read, write, get_session_id).
            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(self.mcp_url)
            )
        # else:
            # Local MCP via stdio .
            # params = StdioServerParameters(command=self.server_command, args=self.server_args)
            # read, write = await self._stack.enter_async_context(stdio_client(params))

        self._session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        self.tools = await load_mcp_tools(session=self._session, format="openai")
        self._system_prompt = self.build_system_prompt(self.tools) # intiate the system prompt with the tools available

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    async def __aenter__(self) -> "WaterpointsAgent":
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    def _build_messages(
        self, user_message: str, trail: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Past turns: user question and final answer only. Current turn: full tool trail."""
        history: list[dict[str, Any]] = []
        for turn in self.turns[-self.max_memory_turns:]:
            history.append({"role": "user", "content": turn["user"]})
            history.append({"role": "assistant", "content": turn["answer"]})

        system_prompt = self._system_prompt or self.build_system_prompt(self.tools)
        history.append({"role": "user", "content": user_message})
        return [system_prompt, *history, *trail]

    @staticmethod
    def build_system_prompt(tools: list[dict[str, Any]]) -> dict[str, str]:
        tools_description = "\n".join(
            (
                f"- {tool['function']['name']}: "
                f"{tool['function'].get('description', 'No description available.')}"
            )
            for tool in tools
        )

        return {
            "role": "system",
            "content": SYSTEM_PROMPT_TEMPLATE.format(
                tools_description=tools_description
            ),
        }

    def reset_memory(self) -> None:
        self.turns.clear()
        _RescuedToolCall._counter = 0

    @staticmethod
    def _clean_answer(text: str | None) -> str:
        t = (text or "").strip()

        if t.startswith("{"):
            try:
                obj = json.loads(t)
                if isinstance(obj, dict):
                    for key in ("answer", "response", "text", "message"):
                        if isinstance(obj.get(key), str):
                            return obj[key].strip()
            except json.JSONDecodeError:
                pass

            m = re.match(r"^\{\s*\"?answer\"?\s*:\s*(.*?)\s*\}$", t, re.S)
            if m:
                return m.group(1).strip().strip('"')

        return t

    async def chat(self, user_message: str) -> str:
        """Process a user message through the LLM and waterpoints MCP tools."""

        if not user_message.strip():
            return "Please provide a non-empty message to process."
        if self._session is None:
            await self.start()

        trail: list[dict[str, Any]] = []
        executed_calls: dict[str, str] = {}
        repeated_call = 0

        for _ in range(self.max_iterations):
            messages = self._build_messages(user_message, trail)
            kwargs: dict[str, Any] = dict(
                model=self.model,
                api_base=self.api_base,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                num_ctx=self.num_ctx,
                keep_alive=self.keep_alive,
                timeout=self.request_timeout,
            )
            if repeated_call == 0:
                kwargs["tools"] = self.tools

            try:
                response = await acompletion(**kwargs)
            except Exception as exc:
                logger.exception("LLM call failed")
                return f"The language model did not respond ({exc}). Check that Ollama is running."

            message = response.choices[0].message
            tool_calls = list(message.tool_calls or [])

            if not tool_calls and message.content and repeated_call == 0:
                rescued = self._extract_text_tool_calls(message.content)
                if rescued:
                    logger.warning("Rescued %s tool call(s) emitted as plain text", len(rescued))
                    print(f"Rescued {len(rescued)} tool call(s) emitted as plain text")
                    tool_calls = rescued
                    message.content = None

            if not tool_calls:
                final_content = self._clean_answer(message.content) or (
                    "No answer could be generated from the tool results."
                )
                self.turns.append({"user": user_message, "answer": final_content})
                return final_content

            trail.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [tool_call.model_dump() for tool_call in tool_calls],
            })
            print(f"Tool calls detected: {len(tool_calls)}")

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_arguments = self._parse_tool_arguments(tool_call.function.arguments)
                call_key = self._build_call_key(tool_name, tool_arguments)
                print("this tool call key is: ", call_key, " is already executed: ", call_key in executed_calls)
                if call_key in executed_calls:
                    logger.warning("Repeated call to %s with %s - serving cached result",tool_name,tool_arguments,)
                    print(f"Repeated call to {tool_name} with {tool_arguments} - serving cached result")
                    repeated_call += 1
                    content = (
                        "[CACHED] This exact call was already made. Result below. "
                        "Do not call it again.\n\n" + executed_calls[call_key]
                    )
                else:
                    result = await self._execute_tool(self._session, tool_name, tool_arguments)
                    content = self._format_tool_result(result)
                    executed_calls[call_key] = content

                trail.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": content,
                })

            if repeated_call: #repeat is true force the final answer to be generated
                trail.append({
                    "role": "system",
                    "content": "Write the final answer now from the results above.",
                })

        fallback = (
            "The question could not be completed within the allowed steps. "
            "Please rephrase it or give the waterpoint name and period."
        )
        self.turns.append({"user": user_message, "answer": fallback})
        return fallback

    @staticmethod
    def _format_tool_result(result: dict[str, Any]) -> str:
        """Turn an MCP tool result dict into plain text for the LLM."""
        if result.get("is_error"):
            err = (
                result.get("error")
                or result.get("content")
                or "Unknown tool error"
            )
            return f"[TOOL ERROR] {err}"

        mcp_response_content = result.get("content", [])

        if isinstance(mcp_response_content, list):
            formatted_items = []
            for item in mcp_response_content:
                if isinstance(item, (dict, list)):
                    formatted_items.append(
                        json.dumps(
                            item, ensure_ascii=False, indent=2, default=str
                        )
                    )
                else:
                    formatted_items.append(str(item))
            plain_text = "\n".join(formatted_items)
        elif isinstance(mcp_response_content, dict):
            plain_text = json.dumps(
                mcp_response_content,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        else:
            plain_text = (
                str(mcp_response_content)
                if mcp_response_content is not None
                else ""
            )

        if not plain_text.strip() and "structured_content" in result:
            plain_text = json.dumps(
                result["structured_content"],
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        if not plain_text.strip():
            return (
                "[EMPTY RESULT] The tool returned no data. "
                "Do not treat this as a zero value. "
                "Tell the user that no matching information was found, "
                "or try different arguments / another tool."
            )

        if len(plain_text) > _MAX_TOOL_CHARS:
            plain_text = (
                plain_text[:_MAX_TOOL_CHARS]
                + "\n...[TRUNCATED — summarize what you have and, if needed, "
                "narrow the query range]"
            )

        return plain_text

    async def _execute_tool(
        self,
        session: ClientSession,
        tool_name: str,
        tool_arguments: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            mcp_result = await session.call_tool(
                name=tool_name,
                arguments=tool_arguments,
            )
            logger.debug("MCP RESULT for %s: %s", tool_name, mcp_result)

            text_blocks = [
                block.text
                for block in mcp_result.content
                if hasattr(block, "text")
            ]

            result: dict[str, Any] = {
                "content": text_blocks,
                "is_error": bool(getattr(mcp_result, "isError", False)),
            }

            structured_content = getattr(
                mcp_result, "structuredContent", None
            )
            if structured_content is not None:
                result["structured_content"] = structured_content

            return result

        except Exception as exc:
            logger.exception("Error executing MCP tool %s", tool_name)
            return {
                "content": [f"Error executing {tool_name}: {exc}"],
                "error": f"Error executing {tool_name}: {exc}",
                "is_error": True,
            }

    @staticmethod
    def _extract_text_tool_calls(content: str) -> list[_RescuedToolCall]:
        """Detect tool calls that llama3.1 emitted as plain-text JSON."""
        text = content.strip()
        if not text.startswith("{"):
            return []

        candidates: list[str] = []
        try:
            json.loads(text)
            candidates = [text]
        except json.JSONDecodeError:
            parts = [
                p.strip().rstrip(";")
                for p in text.replace("};", "}\n").splitlines()
            ]
            candidates = [p for p in parts if p.startswith("{")]

        rescued: list[_RescuedToolCall] = []
        for candidate in candidates:
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                return []
            if not isinstance(obj, dict):
                return []
            name = obj.get("name")
            arguments = obj.get("parameters", obj.get("arguments"))
            if not isinstance(name, str) or not isinstance(arguments, dict):
                return []
            rescued.append(_RescuedToolCall(name=name, arguments=arguments))

        return rescued

    @staticmethod
    def _build_call_key(tool_name: str, tool_arguments: dict[str, Any]) -> str:
        return f"{tool_name}:" + json.dumps(
            tool_arguments,
            sort_keys=True,
            default=str,
        )

    @staticmethod
    def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
        """Return a Python dict only; never raise on malformed model output."""
        if isinstance(arguments, dict):
            return arguments
        if not arguments:
            return {}
        try:
            parsed = json.loads(arguments)
        except (TypeError, json.JSONDecodeError):
            logger.warning(
                "Malformed tool arguments from model, defaulting to {}: %r",
                arguments,
            )
            return {}
        if not isinstance(parsed, dict):
            logger.warning(
                "Tool arguments did not decode to a JSON object, defaulting to {}: %r",
                arguments,
            )
            return {}
        return parsed