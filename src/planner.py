"""
Planner LLM Module

This module is responsible for analyzing user requirements, asking clarification
questions if needed, and generating a structured notebook outline.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any, Union, Callable, cast

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

# Use absolute imports for proper package imports when running from root
from src.models import (
    NotebookPlanModel,
    Section,
)
from src.prompts.planner_prompts import (
    PLANNING_SYSTEM_PROMPT,
    PLANNING_PROMPT,
    PLANNING_WITH_CLARIFICATION_PROMPT,
)

from src.format.prompt_utils import (
    format_additional_requirements,
    format_code_snippets,
    format_clarifications,
)

from src.tools.clarification_tools import get_clarifications_tool
from src.tools.utils import extract_tool_arguments, has_tool_call

# Get a logger for this module
logger = logging.getLogger(__name__)

# Try to import dotenv, but make it optional
try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ImportError:
    # If dotenv is not installed, just print a warning
    print("Warning: python-dotenv not installed. Using environment variables directly.")


class PlannerLLM:
    """
    LLM-based notebook planner.

    This class uses OpenAI models to analyze user requirements and generate
    a structured plan for a Jupyter notebook.
    """

    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the planner with the specified model.

        Args:
            model: The OpenAI model to use for planning
        """
        self.model = model
        self.client = OpenAI()
        logger.info(f"PlannerLLM initialized with model: {model}")

        # Check if API key is set
        if not os.environ.get("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable not set")
            raise ValueError(
                "OPENAI_API_KEY environment variable must be set to use the PlannerLLM"
            )

        # Get the clarification tool
        self.tools: List[ChatCompletionToolParam] = [get_clarifications_tool()]
        logger.info(
            f"Tools configured: {[tool['function']['name'] for tool in self.tools]}"
        )

    def plan_notebook(
        self,
        requirements: Dict[str, Any],
        clarification_callback: Optional[Callable[[List[str]], Dict[str, str]]] = None,
        search_results: Optional[str] = None,
    ) -> NotebookPlanModel:
        """
        Plan a notebook based on the given requirements.

        Args:
            requirements: A dictionary containing the notebook requirements.
            clarification_callback: A callback function to get clarifications from the user.
            search_results: Optional string containing search results about the notebook topic.

        Returns:
            A NotebookPlanModel containing the notebook plan.
        """
        logger.info("Starting plan_notebook method")
        logger.info(f"Received requirements: {requirements}")
        logger.info(f"Has clarification callback: {clarification_callback is not None}")
        logger.info(f"Has search results: {search_results is not None}")

        # Extract requirements
        description = requirements.get("notebook_description", "Not provided")
        purpose = requirements.get("notebook_purpose", "Not provided")
        target_audience = requirements.get("target_audience", "Not provided")
        code_snippets = requirements.get("code_snippets", [])
        additional_reqs = requirements.get("additional_requirements", [])

        # Format the requirements for the prompt
        additional_requirements = format_additional_requirements(additional_reqs)
        formatted_code_snippets = format_code_snippets(code_snippets)
        formatted_search_results = "No search results available"
        if search_results:
            formatted_search_results = search_results

        # Start with the initial formatted prompt
        system_prompt = PLANNING_SYSTEM_PROMPT
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
        ]

        # Add the user prompt with or without search results
        user_prompt = PLANNING_PROMPT.format(
            description=description,
            purpose=purpose,
            target_audience=target_audience,
            additional_requirements=additional_requirements,
            code_snippets=formatted_code_snippets,
            search_results=formatted_search_results,
        )
        messages.append({"role": "user", "content": user_prompt})

        # Store all clarifications gathered during the process
        all_clarifications = {}

        # Flag to determine if we should continue asking for clarifications
        continue_clarifying = True
        max_clarification_rounds = 3
        rounds = 0

        # Continue until we have no more clarification questions or reach max rounds
        while continue_clarifying and rounds < max_clarification_rounds:
            rounds += 1
            logger.info(
                f"Starting clarification round {rounds}/{max_clarification_rounds}"
            )

            try:
                # Make the API call with tools
                logger.info("Making API call with tools...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    stream=False,  # Ensure we're not using streaming to avoid the choices attribute error
                )

                message = response.choices[0].message
                logger.info(f"Received response from model")

                # Check if the model wants to use the clarification tool
                has_clarification_call = has_tool_call(message, "get_clarifications")
                logger.info(f"Has clarification tool call: {has_clarification_call}")

                if (
                    has_clarification_call
                    and message.tool_calls
                    and clarification_callback
                ):
                    # Since we only have one tool, we can simplify by directly accessing the first tool call
                    tool_call = message.tool_calls[0]
                    args = extract_tool_arguments(tool_call)
                    questions = args.get("questions", [])
                    logger.info(f"Extracted {len(questions)} questions from tool call")

                    if questions:
                        # Get answers from user via the callback
                        logger.info(
                            f"Asking user {len(questions)} clarification questions"
                        )
                        clarification_answers = clarification_callback(questions)

                        # Add to our accumulated clarifications
                        all_clarifications.update(clarification_answers)
                        logger.info(f"Updated all_clarifications with new answers")

                        # Add this interaction to the message history
                        messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": tool_call.id,
                                        "type": "function",
                                        "function": {
                                            "name": tool_call.function.name,
                                            "arguments": tool_call.function.arguments,
                                        },
                                    }
                                ],
                            }
                        )

                        # Add the tool response to the message history
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(clarification_answers),
                            }
                        )

                        # Create a new prompt with the clarifications
                        clarification_prompt = (
                            PLANNING_WITH_CLARIFICATION_PROMPT.format(
                                description=description,
                                purpose=purpose,
                                target_audience=target_audience,
                                additional_requirements=additional_requirements,
                                code_snippets=formatted_code_snippets,
                                clarifications=format_clarifications(
                                    all_clarifications
                                ),
                                search_results=formatted_search_results,
                            )
                        )

                        # Add this new prompt to continue the conversation
                        messages.append(
                            {
                                "role": "user",
                                "content": clarification_prompt,
                            }
                        )
                        logger.info(
                            "Added clarification-enhanced prompt to message history"
                        )
                    else:
                        logger.info(
                            "No questions found in tool call, ending clarification rounds"
                        )
                        continue_clarifying = False
                else:
                    # No more clarification questions or no callback, proceed to final plan generation
                    logger.info(
                        "No clarification needed, proceeding to plan generation"
                    )
                    continue_clarifying = False

                    # Use structured output to get the notebook plan
                    logger.info("Generating notebook plan with structured output...")
                    notebook_plan = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        response_format=NotebookPlanModel,
                    )

                    parsed_plan = notebook_plan.choices[0].message.parsed
                    if parsed_plan is not None:
                        logger.info("Successfully parsed notebook plan")
                        return parsed_plan

                    # If parsed result is None, create an error plan
                    logger.info("Failed to parse notebook plan, returning error plan")
                    return self._create_error_plan("Failed to parse notebook plan")

            except Exception as e:
                logger.info(f"Error in plan_notebook: {e}")
                return self._create_error_plan(str(e))

        # Final attempt to generate plan with all accumulated clarifications
        if all_clarifications:
            logger.info(
                f"Final plan generation with {len(all_clarifications)} clarifications"
            )
            formatted_prompt_with_clarifications = (
                PLANNING_WITH_CLARIFICATION_PROMPT.format(
                    description=requirements.get(
                        "notebook_description", "Not provided"
                    ),
                    purpose=requirements.get("notebook_purpose", "Not provided"),
                    target_audience=requirements.get("target_audience", "Not provided"),
                    additional_requirements=format_additional_requirements(
                        requirements.get("additional_requirements", [])
                    ),
                    code_snippets=format_code_snippets(
                        requirements.get("code_snippets", [])
                    ),
                    clarifications=format_clarifications(all_clarifications),
                )
            )

            # Final request for notebook plan using structured output
            try:
                logger.info("Making final API call for notebook plan...")
                notebook_plan = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": formatted_prompt_with_clarifications,
                        },
                    ],
                    response_format=NotebookPlanModel,
                )

                parsed_plan = notebook_plan.choices[0].message.parsed
                if parsed_plan is not None:
                    logger.info("Successfully parsed final notebook plan")
                    return parsed_plan

                # If parsed result is None, create an error plan
                logger.info("Failed to parse final notebook plan, returning error plan")
                return self._create_error_plan("Failed to parse notebook plan")

            except Exception as e:
                logger.info(f"Error in final plan generation: {e}")

        # If we reach here, try to generate a plan from the last state
        try:
            logger.info(
                "Attempting fallback plan generation from last message state..."
            )
            notebook_plan = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=NotebookPlanModel,
            )

            parsed_plan = notebook_plan.choices[0].message.parsed
            if parsed_plan is not None:
                logger.info("Successfully parsed fallback notebook plan")
                return parsed_plan

            # If parsed result is None, create an error plan
            logger.info("Failed to parse fallback notebook plan, returning error plan")
            return self._create_error_plan("Failed to parse notebook plan")

        except Exception as e:
            logger.info(f"Error in fallback plan generation: {e}")

        # Return error plan if all attempts fail
        logger.info("All plan generation attempts failed, returning error plan")
        return self._create_error_plan("Failed to generate notebook plan")

    def _create_error_plan(self, error_message: str) -> NotebookPlanModel:
        """Create a basic error plan when notebook generation fails."""
        logger.info(f"Creating error plan: {error_message}")
        return NotebookPlanModel(
            title="Error generating plan",
            description="There was an error generating the notebook plan.",
            purpose="N/A",
            target_audience="N/A",
            sections=[
                Section(
                    title="Error",
                    description=f"Error details: {error_message}",
                    subsections=None,
                )
            ],
        )

    def plan_notebook_with_ui(
        self,
        requirements: Dict[str, Any],
        previous_messages: Optional[List[Dict[str, Any]]] = None,
        clarification_answers: Optional[Dict[str, str]] = None,
        search_results: Optional[str] = None,
    ) -> Union[NotebookPlanModel, List[str]]:
        """
        Plan a notebook with UI-compatible clarification handling.

        This method is designed to work with UIs like Streamlit that need to handle
        async operations through multiple reruns rather than callbacks.

        Args:
            requirements: A dictionary containing the notebook requirements.
            previous_messages: Previous chat messages if continuing a conversation.
            clarification_answers: Answers to previous clarification questions.
            search_results: Optional string containing search results about the notebook topic.

        Returns:
            Either a NotebookPlanModel if planning is complete, or a List[str] of
            clarification questions if user input is needed.
        """
        logger.info("Starting plan_notebook_with_ui method")
        logger.info(f"Received requirements: {requirements}")
        logger.info(f"Has previous messages: {previous_messages is not None}")
        logger.info(f"Has clarification answers: {clarification_answers is not None}")
        logger.info(f"Has search results: {search_results is not None}")

        # Extract requirements
        description = requirements.get("notebook_description", "Not provided")
        purpose = requirements.get("notebook_purpose", "Not provided")
        target_audience = requirements.get("target_audience", "Not provided")
        code_snippets = requirements.get("code_snippets", [])
        additional_reqs = requirements.get("additional_requirements", [])

        # Format the requirements for the prompt
        additional_requirements = format_additional_requirements(additional_reqs)
        formatted_code_snippets = format_code_snippets(code_snippets)
        formatted_search_results = "No search results available"
        if search_results:
            formatted_search_results = search_results

        # Initialize or use provided message history
        system_prompt = PLANNING_SYSTEM_PROMPT

        # Initialize messages as List[Dict[str, Any]] to handle all message types
        messages: List[Dict[str, Any]] = []
        # Initialize api_messages to be used in the API call
        api_messages: List[ChatCompletionMessageParam] = []

        if previous_messages:
            messages = previous_messages
            logger.info(f"Using {len(messages)} previous messages")
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": PLANNING_PROMPT.format(
                        description=description,
                        purpose=purpose,
                        target_audience=target_audience,
                        additional_requirements=additional_requirements,
                        code_snippets=formatted_code_snippets,
                        search_results=formatted_search_results,
                    ),
                },
            ]
            logger.info("Created new message history")

        # Store messages in the instance for later retrieval
        self._current_messages = messages

        # If we have clarification answers from the UI, add them to the messages
        if clarification_answers and len(messages) >= 3:
            # Get the assistant message that should contain tool calls
            assistant_message = messages[-2]
            if (
                isinstance(assistant_message, dict)
                and assistant_message.get("role") == "assistant"
            ):
                tool_calls = assistant_message.get("tool_calls", [])
                if isinstance(tool_calls, list) and tool_calls:
                    tool_call = tool_calls[0]
                    if isinstance(tool_call, dict):
                        tool_call_id = tool_call.get("id")

                        if tool_call_id:
                            # Remove any existing placeholder tool response
                            messages = [
                                msg
                                for msg in messages
                                if not (
                                    isinstance(msg, dict)
                                    and msg.get("role") == "tool"
                                    and msg.get("tool_call_id") == tool_call_id
                                )
                            ]

                            # Add the tool response message with actual answers
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": json.dumps(clarification_answers),
                                }
                            )

                            # Add a new user message with a prompt that includes the clarifications
                            all_clarifications = clarification_answers  # In this case we only have the current answers

                            clarification_prompt = (
                                PLANNING_WITH_CLARIFICATION_PROMPT.format(
                                    description=description,
                                    purpose=purpose,
                                    target_audience=target_audience,
                                    additional_requirements=additional_requirements,
                                    code_snippets=formatted_code_snippets,
                                    clarifications=format_clarifications(
                                        all_clarifications
                                    ),
                                    search_results=formatted_search_results,
                                )
                            )

                            messages.append(
                                {
                                    "role": "user",
                                    "content": clarification_prompt,
                                }
                            )

                            logger.info(
                                "Added tool response and new user prompt with clarifications"
                            )

                            # Update stored messages
                            self._current_messages = messages

        # Convert messages to API format
        api_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "")
            content = msg.get("content", "")

            # Handle tool response messages separately
            if role == "tool":
                api_msg: ChatCompletionMessageParam = {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": msg["tool_call_id"],
                }
            else:
                api_msg: ChatCompletionMessageParam = {
                    "role": role,
                    "content": content,
                }
                # Add tool_calls if present
                if "tool_calls" in msg:
                    api_msg["tool_calls"] = msg["tool_calls"]

            api_messages.append(api_msg)

        try:
            # Make the API call with tools
            logger.info("Making API call with tools...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                tools=self.tools,
                stream=False,
            )

            message = response.choices[0].message
            logger.info(f"Received response from model")

            # Check if the model wants to use the clarification tool
            has_clarification_call = has_tool_call(message, "get_clarifications")
            logger.info(f"Has clarification tool call: {has_clarification_call}")

            if has_clarification_call and message.tool_calls:
                # Extract the questions
                tool_call = message.tool_calls[0]
                args = extract_tool_arguments(tool_call)
                questions = args.get("questions", [])

                if questions:
                    logger.info(f"Extracted {len(questions)} questions")

                    # Add the assistant message to the history as Dict[str, Any]
                    new_message: Dict[str, Any] = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                        ],
                    }
                    messages.append(new_message)

                    # Add a placeholder tool response message to satisfy the API requirement
                    tool_response_message: Dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(
                            {}
                        ),  # Empty response since we're waiting for user input
                    }
                    messages.append(tool_response_message)

                    # Update stored messages
                    self._current_messages = messages

                    # Return the questions for the UI to handle
                    return questions

            # No clarification needed, generate the notebook plan
            logger.info("No clarification needed, generating final plan")
            notebook_plan = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=api_messages,
                response_format=NotebookPlanModel,
            )

            parsed_plan = notebook_plan.choices[0].message.parsed
            if parsed_plan is not None:
                logger.info("Successfully parsed notebook plan")
                return parsed_plan

            # If parsed result is None, create an error plan
            logger.info("Failed to parse notebook plan, returning error plan")
            return self._create_error_plan("Failed to parse notebook plan")

        except Exception as e:
            logger.error(f"Error in plan_notebook_with_ui: {e}")
            return self._create_error_plan(str(e))

    def get_current_messages(self) -> List[Dict[str, Any]]:
        """
        Get the current message history.

        Returns:
            The current message history as a list of message dictionaries.
        """
        if hasattr(self, "_current_messages"):
            return self._current_messages
        return []
