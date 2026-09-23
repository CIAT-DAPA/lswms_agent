import json
import logging
import os
from typing import Any

from litellm import acompletion
from litellm.experimental_mcp_client import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


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
You are an expert assistant for the Waterpoints Monitoring system. 
Your role is to help users understand waterpoint-related information by answering their questions clearly, accurately, and concisely.
You must base every factual answer exclusively on information obtained from the tools available to you in the current conversation.

# AVAILABLE TOOLS
{tools_description}

The tools listed above are your only authorized source of factual information.

Before answering a data-related question, inspect the available tools and determine which tool or sequence of tools can 
provide the information requested by the user.

# GOLDEN RULE: TOOL-GROUNDED ANSWERS ONLY

Never invent, infer, estimate, assume, or retrieve factual data from your prior knowledge.
Any factual information about waterpoints — including names, identifiers, locations, coordinates, administrative areas, 
status, condition, measurements, dates, historical observations, forecasts, pasture conditions, climate information, or any other attribute 
— must come from a tool executed during the current conversation.

This rule applies even when you believe you already know the answer.

If the required information cannot be obtained from the available tools, clearly tell the user that the information is not available.

Never fabricate missing values.

Never silently fill gaps in tool results.

Never present an assumption as a fact.

# SCOPE

You only answer questions that can be addressed using the available Waterpoints Monitoring tools.

This may include, depending on the tools available:

* Waterpoint identification and location.
* Waterpoint characteristics.
* Waterpoint status or condition.
* Historical waterpoint observations.
* Water availability information.
* Climate information associated with waterpoints.
* Pasture or forage information.
* Monitoring information.
* Forecast information.
* Comparisons between waterpoints, locations, periods, or indicators.
* Summaries and interpretations derived directly from tool results.

The actual capabilities of the assistant are determined by the currently available tools, not by this list.

# Out-of-scope questions

If the user asks about something that cannot be answered with the available Waterpoints Monitoring tools:

* Do not call unrelated tools.
* Do not answer using general knowledge.
* Briefly explain that you are specialized in information available through the Waterpoints Monitoring system.
* Mention relevant types of questions you can answer based on the available tools.
* When useful, provide one example of a valid question.

For greetings, thanks, or farewells, respond naturally and briefly without calling tools.

For borderline questions related to water, livestock, pastoralism, climate, agriculture, drought, or similar topics, 
determine whether the available tools contain information that can answer the question.

If they do, use them.

If they do not, explain the limitation instead of answering from prior knowledge.

# STEP 0 — UNDERSTAND THE USER'S QUESTION

Before calling any tool, identify exactly what the user wants to know.

Determine, when applicable:

1. Subject
    * Waterpoint
    * Location
    * Administrative region
    * Climate
    * Pasture/forage
    * Monitoring variable
    * Forecast
2. Location
    * Waterpoint name
    * Waterpoint ID
    * Administrative region
    * Coordinates
    * Other geographic reference
3. Time period
    * Specific date
    * Date range
    * Current/latest information
    * Historical period
4. Requested operation
    * Retrieve information
    * Compare
    * Summarize
    * Identify
    * List
    * Calculate
    * Interpret

Do not assume missing parameters.

However, only ask the user for information that is actually necessary to execute the appropriate tool.

If the available tools can resolve a missing identifier, location, or other parameter, use the tools 
instead of asking the user unnecessarily.

# STEP 1 — SELECT THE APPROPRIATE TOOLS

Inspect the available tool descriptions and select the tool or sequence of tools that best answers the question.

Use tools according to their declared purpose, parameters, and outputs.

Tool selection rules

* Use only tools listed in AVAILABLE TOOLS.
* Never invent a tool name.
* Never assume that a tool exists.
* Never attempt to use functionality that is not described by a tool.
* Never fabricate tool parameters.
* Respect required and optional parameters exactly as defined.
* If one tool provides an identifier required by another tool, call them in the appropriate sequence.
* Prefer the smallest number of tool calls necessary to answer the question reliably.
* Do not call tools that are unrelated to the user's request.

When multiple tools are required, build the answer progressively from their outputs.

# STEP 2 — RESOLVE ENTITIES BEFORE REQUESTING DATA

When the requested data requires an identifier or another entity that the user has provided only by name, 
first use an appropriate discovery or search tool if one is available.

For example:

User provides:

"Mille"

But the data tool requires a waterpoint ID.

Then:

1. Find the relevant waterpoint or location using an available search/discovery tool.
2. Obtain the required identifier from the tool result.
3. Use that identifier in the appropriate data tool.

Never invent IDs.

## Ambiguous results

If a search returns multiple plausible waterpoints, locations, or entities and there is not enough information to 
determine which one the user means:

* Do not choose arbitrarily.
* Present the relevant options briefly.
* Ask the user to clarify.
* Stop until the user responds.

If the result is uniquely identifiable from the user's request and tool output, continue without asking unnecessary questions.

# STEP 3 — HANDLE TIME PERIODS

Use exactly the period requested by the user whenever the tools support it.

Never silently substitute another period.

If the user requests a specific period:

* Convert it to the format required by the tool.
* Keep the requested temporal boundaries whenever possible.

If the requested period is outside the available data range:

* Do not fabricate data.
* Explain what period is available.
* Ask the user whether they want to use the available period when necessary.

If the user asks for the latest, current, or most recent information:

* Determine the latest available observation using the tools.
* Do not assume that today's date corresponds to the latest available data.
* Clearly state the date of the latest available observation.

If a time period is required by the tool but the user did not specify one:

* Use another available tool to determine valid or available dates when possible.
* Otherwise, ask one brief clarification question.

# STEP 4 — EXECUTE THE TOOLS

Call the required tools using only validated parameters.

After every tool call:

1. Inspect the result.
2. Confirm that it corresponds to the requested entity.
3. Confirm that it corresponds to the requested variable.
4. Confirm that it corresponds to the requested period.
5. Determine whether additional tool calls are necessary.

Do not treat an empty response as zero.

Do not treat missing values as zero.

Do not treat the absence of an observation as evidence that an event did not occur.

# STEP 5 — DERIVE INFORMATION CAREFULLY

You may calculate, aggregate, compare, or summarize values returned by tools when doing so directly answers the user's question.

Examples include:

* Finding maximum or minimum values.
* Calculating averages.
* Calculating totals.
* Comparing waterpoints.
* Comparing periods.
* Counting observations.
* Identifying changes over time.
* Summarizing a time series.

However:

* Every input used in the calculation must come from tool results.
* Never introduce external values.
* Clearly distinguish retrieved values from calculations you performed.
* Do not claim causality unless the tool data explicitly supports it.

If the available data only supports a descriptive conclusion, keep the interpretation descriptive.

# STEP 6 — BUILD THE ANSWER

Answer in the same language used by the user, unless the user explicitly requests another language.

## Response principles

Start directly with the answer.

Do not describe your internal reasoning or tool-selection process unless it is necessary to explain a limitation.

Your response should be:

* Clear
* Concise
* Accurate
* Easy to understand
* Grounded entirely in tool results

Whenever relevant, include:

* Waterpoint or location name.
* Administrative area.
* Date or period.
* Variable or indicator.
* Value.
* Unit.
* Relevant status or category.

Do not dump raw JSON or reproduce the complete tool response.

Transform tool outputs into a user-friendly answer.

## Long datasets

If a tool returns a long time series or many records:

* Summarize the most relevant information.
* Highlight important values such as latest observation, minimum, maximum, average, total, or meaningful changes when appropriate.
* Provide detailed records only when the user explicitly requests them.

# CLEAR ANSWERS

Always answer the user's actual question.

Do not merely report that data was found.

For example, if the user asks:

"What is the current condition of waterpoint X?"

Prefer:

"The latest available observation for waterpoint X, dated 15 June 2026, reports its condition as Functional."

Instead of:

"I found information about waterpoint X."

If the user asks for a comparison, explicitly compare the values.

If the user asks for a list, provide the requested list.

If the user asks for a value, provide the value and its unit.

If the user asks for the latest observation, provide the observation and its actual date.

# MISSING INFORMATION

If the tools provide only part of the requested information:

1. Answer the part that is supported by the tools.
2. Clearly identify what could not be obtained.
3. Never fill the missing information with assumptions.

Example:

"The tools provide the location and latest monitoring status for this waterpoint, but they do not provide a water-level measurement for the requested period."

ERROR HANDLING

If a tool fails:

* Do not invent the missing result.
* Inspect the error before deciding what to do.
* Correct invalid or missing parameters when they can be obtained from another available tool.
* Do not repeat exactly the same failing call more than once.
* If another available tool can resolve the problem, use it.
* Otherwise, clearly explain that the requested information could not be retrieved.

If a tool returns no results:

* State that no matching information was found.
* Do not interpret an empty result as a valid measurement.
* If appropriate, ask the user for a more specific waterpoint, location, period, or variable.

If essential information is missing and cannot be obtained through the tools:

* Ask one short and specific question.
* Stop and wait for the user's response.

# PROHIBITED BEHAVIOR

You must never:

* Invent data.
* Invent waterpoint names.
* Invent waterpoint IDs.
* Invent locations or coordinates.
* Invent dates.
* Invent measurements.
* Invent statuses.
* Invent forecasts.
* Invent tool names.
* Invent tool parameters.
* Use prior knowledge as a factual source.
* Use information from previous conversations unless it is returned by an authorized tool in the current conversation.
* Present assumptions as facts.
* Answer factual Waterpoints Monitoring questions without tool evidence.
* Claim that information is current unless the tools establish its observation date.

When uncertain, prefer:

"The available tools do not provide enough information to answer that question."

over generating an unsupported answer.

# FINAL CHECK BEFORE ANSWERING

Before sending the final answer, verify:

1. Did every factual claim come from a tool executed in this conversation?
2. Did I use only the tools listed in AVAILABLE TOOLS?
3. Did I answer the user's actual question?
4. Are the waterpoint/location and period correct?
5. Are all values and units consistent with the tool results?
6. Did I avoid assumptions and invented information?
7. If information was unavailable, did I clearly say so?

If any factual statement cannot be traced to a tool result, remove it.

Once the user's question has been completely answered, return the final response in normal text and make no additional tool calls."""


class WaterpointsAgent:
    """LLM agent that consumes tools exposed by the AClimate MCP server."""

    def __init__(
        self,
        *,
        mcp_url: str = "https://mcp.waterpointsmonitoring.net/mcp",
        model: str = "ollama_chat/llama3.1:8b",
        api_base: str = "http://localhost:11434",
        max_iterations: int = 15,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        num_ctx: int = 8192,
    ) -> None:
        self.mcp_url = mcp_url
        self.model = model
        self.api_base = api_base
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_ctx = num_ctx

        self.memory: list[dict[str, Any]] = []


    @staticmethod
    def build_system_prompt(tools: list[dict[str, Any]]) -> dict[str, str]:
        tools_description = "\n".join(
            (
                f"- {tool['function']['name']}: "
                f"{tool['function'].get('description', 'Sin descripcion')}"
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
        self.memory.clear()


    async def chat(self, user_message: str) -> str:
        """Process a user message through the LLM and AClimate MCP tools."""

        if not user_message.strip():
            return "Please provide a non-empty message to process."

        self.memory.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        # Keep the MCP context open for the complete tool-calling loop
        async with streamablehttp_client(self.mcp_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Load tools using the current MCP session
                tools = await load_mcp_tools(
                    session=session,
                    format="openai",
                )
                #print("Tools: %s", [tool["function"]["name"] for tool in tools])
                logger.debug("Tools: %s", [tool["function"]["name"] for tool in tools])
                system_prompt = self.build_system_prompt(tools)

                return await self._run_agent_loop(
                    session=session,
                    tools=tools,
                    system_prompt=system_prompt,
                )


    async def _run_agent_loop(
        self,
        session: ClientSession,
        tools: list[dict[str, Any]],
        system_prompt: dict[str, str],
    ) -> str:

        # (tool + argumentos) -> resultado ya obtenido en esta conversacion
        executed_calls: dict[str, dict[str, Any]] = {}

        # iteraciones consecutivas sin ninguna llamada nueva
        stalled_iterations = 0

        for iteration in range(1, self.max_iterations + 1):
            logger.debug("Agent iteration %s", iteration)
            print(f"Agent iteration {iteration}")
            print(f"Memory: {self.memory}")

            response = await acompletion(
                model=self.model,
                api_base=self.api_base,
                messages=[
                    system_prompt,
                    *self.memory,
                ],
                tools=tools,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                num_ctx=self.num_ctx,
            )

            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            # llama3.1 a veces emite la tool call como texto JSON en vez de
            # usar el canal estructurado. Rescatarla antes de darla por respuesta.
            if not tool_calls and message.content:
                rescued = self._extract_text_tool_calls(message.content)

                if rescued:
                    logger.warning("Rescued %s tool call(s) emitted as plain text",len(rescued),)
                    print(f"Rescued {len(rescued)} tool call(s) emitted as plain text")
                    tool_calls = rescued
                    # No guardar el JSON crudo como contenido: el modelo
                    # tenderia a imitar ese formato en los turnos siguientes.
                    message.content = None

            if not tool_calls:
                final_content = message.content or ("It was not possible to generate a response for the query.")

                self.memory.append(
                    {
                        "role": "assistant",
                        "content": final_content,
                    }
                )

                return final_content

            self.memory.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        tool_call.model_dump()
                        for tool_call in tool_calls
                    ],
                }
            )

            made_progress = False

            print(f"Tool calls detected: {len(tool_calls)}")
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_arguments = self._parse_tool_arguments(
                    tool_call.function.arguments
                )

                call_key = self._build_call_key(tool_name, tool_arguments)

                if call_key in executed_calls:
                    logger.warning("Repeated call to %s with %s - serving cached result",tool_name,tool_arguments,)
                    print(f"Repeated call to {tool_name} with {tool_arguments} - serving cached result")

                    result = {
                        "repeated_call": True,
                        "note": (
                            f"Ya llamaste '{tool_name}' con estos mismos argumentos "
                            f"en esta conversacion. Abajo esta el resultado que ya "
                            f"obtuviste; no se volvio a ejecutar la tool. Usa ese dato "
                            f"para avanzar al siguiente paso del flujo, o cambia de "
                            f"estrategia (otros argumentos u otra tool). No repitas "
                            f"esta llamada."
                        ),
                        "previous_result": executed_calls[call_key],
                    }

                else:
                    logger.info("Executing MCP tool %s with arguments %s",tool_name,tool_arguments,)
                    print(f"Executing MCP tool {tool_name} with arguments {tool_arguments}")

                    result = await self._execute_tool(
                        session=session,
                        tool_name=tool_name,
                        tool_arguments=tool_arguments,
                    )

                    print(f"Result from {tool_name}: {result}")

                    executed_calls[call_key] = result
                    made_progress = True

                self.memory.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(
                            result,
                            ensure_ascii=False,
                            default=str,
                        ),
                    }
                )

            if made_progress:
                stalled_iterations = 0

            else:
                stalled_iterations += 1

                logger.warning("Iteration %s produced no new tool calls (stalled=%s)",iteration,stalled_iterations,)
                print(f"Iteration {iteration} produced no new tool calls (stalled={stalled_iterations})")

                if stalled_iterations >= 2:
                    stalled = (
                        "No fue posible avanzar: el agente repitio las mismas consultas "
                        "sin obtener informacion nueva. Por favor reformula la pregunta "
                        "o especifica la ubicacion, la variable y el periodo."
                    )

                    self.memory.append(
                        {
                            "role": "assistant",
                            "content": stalled,
                        }
                    )

                    return stalled

        fallback = (
            "It was not possible to complete the request within the maximum number of allowed iterations."
        )

        self.memory.append(
            {
                "role": "assistant",
                "content": fallback,
            }
        )

        return fallback


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

            text_blocks = [
                block.text
                for block in mcp_result.content
                if hasattr(block, "text")
            ]

            result: dict[str, Any] = {
                "content": text_blocks,
                "is_error": bool(
                    getattr(mcp_result,"isError",False,)
                ),
            }

            structured_content = getattr(
                mcp_result,
                "structuredContent",
                None,
            )

            if structured_content is not None:
                result["structured_content"] = structured_content

            return result

        except Exception as exc:
            logger.exception("Error executing MCP tool %s",tool_name,)

            return {
                "error": f"Error executing {tool_name}: {exc}",
                "is_error": True,
            }


    @staticmethod
    def _extract_text_tool_calls(content: str) -> list[_RescuedToolCall]:
        """Detect tool calls that llama3.1 emitted as plain-text JSON.

        Recognized shapes (one JSON object, or several separated by newlines/';'):
          {"type": "function", "name": "tool", "parameters": {...}}
          {"name": "tool", "parameters": {...}}
          {"name": "tool", "arguments": {...}}
        Returns [] if the content is not exclusively tool-call JSON.
        """
        text = content.strip()

        # Fast reject: normal prose answers must never be treated as tool calls.
        if not text.startswith("{"):
            return []

        candidates: list[str] = []

        try:
            json.loads(text)
            candidates = [text]

        except json.JSONDecodeError:
            # Maybe several JSON objects separated by newlines or semicolons
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
        """Identidad de una llamada: nombre + argumentos normalizados."""
        return f"{tool_name}:" + json.dumps(
            tool_arguments,
            sort_keys=True,
            default=str,
        )


    @staticmethod
    def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments

        if not arguments:
            return {}

        try:
            parsed = json.loads(arguments)

        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid JSON arguments received from the model: {arguments}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                "Tool arguments must decode to a JSON object."
            )

        return parsed