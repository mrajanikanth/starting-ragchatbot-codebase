import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for searching course information.

Tool Usage:
- **`get_course_outline`**: Use for outline, structure, or overview questions — "what lessons does X have?", "list the topics in X", "give me an overview of X". Return the course title, course link, and every lesson with its number and title.
- **`search_course_content`**: Use for questions about specific content or concepts within a course.
- **Up to 2 sequential tool calls per query** — use a second tool call only when the first result is genuinely insufficient to answer the question
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using a tool
- **Course outline/structure questions**: Call `get_course_outline`, then present the title, course link, and full numbered lesson list
- **Course-specific content questions**: Call `search_course_content`, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    @staticmethod
    def _extract_text(response) -> str:
        """Return the text of the first TextBlock in the response content."""
        for block in response.content:
            if block.type == "text":
                return block.text
        raise ValueError(f"No text block found in response (stop_reason={response.stop_reason!r})")

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )
        messages = [{"role": "user", "content": query}]

        if tools and tool_manager:
            return self._run_agentic_loop(messages, system_content, tools, tool_manager)

        response = self.client.messages.create(
            **self.base_params, messages=messages, system=system_content
        )
        return self._extract_text(response)

    def _run_agentic_loop(self, messages: list, system_content: str,
                          tools: list, tool_manager) -> str:
        """
        Run up to MAX_TOOL_ROUNDS of tool-calling, then return a text response.

        Each round makes an API call with tools; if Claude requests a tool, it is
        executed and results are appended before the next round. Terminates early
        when Claude returns a text response. If all rounds are exhausted, a final
        no-tools API call forces a text answer.
        """
        for _ in range(self.MAX_TOOL_ROUNDS):
            response = self.client.messages.create(
                **self.base_params,
                messages=messages,
                system=system_content,
                tools=tools,
                tool_choice={"type": "auto"},
            )

            if response.stop_reason != "tool_use":
                return self._extract_text(response)

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = tool_manager.execute_tool(block.name, **block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        # MAX_TOOL_ROUNDS exhausted — force a text response without tools
        final_response = self.client.messages.create(
            **self.base_params,
            messages=messages,
            system=system_content,
        )
        return self._extract_text(final_response)