"""
Planner LLM Module

This module is responsible for analyzing user requirements, asking clarification
questions if needed, and generating a structured notebook outline.
"""

import json
import os
from typing import Dict, List, Optional, Any, Union, Callable, cast

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

# Use absolute imports for proper package imports when running from root
from src.models import (
    ClarificationQuestions,
    NotebookPlanModel,
    Section,
    SubSection,
    SubSubSection,
)
from src.prompts.planner_prompts import (
    PLANNING_SYSTEM_PROMPT,
    PLANNING_PROMPT,
    PLANNING_WITH_CLARIFICATION_PROMPT,
    format_additional_requirements,
    format_code_snippets,
    format_clarifications,
)
from src.tools.clarification_tools import get_clarifications_tool
from src.tools.utils import extract_tool_arguments, has_tool_call

# Debug flag - set to True for additional debug output
DEBUG = True


def debug_print(message: str):
    """Print debug message if DEBUG is True."""
    if DEBUG:
        print(f"[DEBUG] {message}")


# Try to import dotenv, but make it optional
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # If dotenv is not installed, just print a warning
    print("Warning: python-dotenv not installed. Using environment variables directly.")


class PlannerLLM:
    """
    Handles the planning phase of notebook generation.

    This class is responsible for analyzing user requirements and generating
    a structured notebook outline, with the ability to ask clarification questions.
    """

    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the PlannerLLM.

        Args:
            model: The OpenAI model to use for planning
        """
        self.model = model
        self.client = OpenAI()

        # Get the clarification tool
        self.tools: List[ChatCompletionToolParam] = [get_clarifications_tool()]
        debug_print(f"Initialized PlannerLLM with model: {model}")
        debug_print(
            f"Tools configured: {[tool['function']['name'] for tool in self.tools]}"
        )

    def plan_notebook(
        self,
        requirements: Dict[str, Any],
        clarification_callback: Optional[Callable[[List[str]], Dict[str, str]]] = None,
    ) -> NotebookPlanModel:
        """
        Plan a notebook based on user requirements.

        If clarification questions are needed, it will use the provided callback
        to get answers from the user.

        Args:
            requirements: User requirements for the notebook
            clarification_callback: Optional callback function to get answers to clarification questions

        Returns:
            A NotebookPlanModel object
        """
        debug_print("Starting plan_notebook method")
        debug_print(f"Received requirements: {requirements}")
        debug_print(f"Has clarification callback: {clarification_callback is not None}")

        # Format the initial prompt for planning
        formatted_prompt = PLANNING_PROMPT.format(
            description=requirements.get("notebook_description", "Not provided"),
            purpose=requirements.get("notebook_purpose", "Not provided"),
            target_audience=requirements.get("target_audience", "Not provided"),
            additional_requirements=format_additional_requirements(
                requirements.get("additional_requirements", [])
            ),
            code_snippets=format_code_snippets(requirements.get("code_snippets", [])),
        )

        # Start with the initial formatted prompt
        system_prompt = PLANNING_SYSTEM_PROMPT
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_prompt},
        ]

        # Store all clarifications gathered during the process
        all_clarifications = {}

        # Flag to determine if we should continue asking for clarifications
        continue_clarifying = True
        max_clarification_rounds = 3
        rounds = 0

        # Continue until we have no more clarification questions or reach max rounds
        while continue_clarifying and rounds < max_clarification_rounds:
            rounds += 1
            debug_print(
                f"Starting clarification round {rounds}/{max_clarification_rounds}"
            )

            try:
                # Make the API call with tools
                debug_print("Making API call with tools...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                )

                message = response.choices[0].message
                debug_print(f"Received response from model")

                # Check if the model wants to use the clarification tool
                has_clarification_call = has_tool_call(message, "get_clarifications")
                debug_print(f"Has clarification tool call: {has_clarification_call}")

                if has_clarification_call:
                    # Only proceed if we have a callback for clarifications
                    if not clarification_callback:
                        debug_print(
                            "No clarification callback provided, skipping clarification"
                        )
                        break

                    # Process each tool call
                    if (
                        message.tool_calls
                    ):  # Ensure tool_calls is not None before iterating
                        debug_print(f"Found {len(message.tool_calls)} tool calls")
                        for tool_call in message.tool_calls:
                            debug_print(
                                f"Processing tool call: {tool_call.function.name}"
                            )
                            if tool_call.function.name == "get_clarifications":
                                args = extract_tool_arguments(tool_call)
                                questions = args.get("questions", [])
                                debug_print(
                                    f"Extracted {len(questions)} questions from tool call"
                                )

                                if questions:
                                    # Get answers from user via the callback
                                    debug_print(
                                        "Calling clarification callback with questions..."
                                    )
                                    new_clarifications = clarification_callback(
                                        questions
                                    )
                                    debug_print(
                                        f"Received {len(new_clarifications)} answers from callback"
                                    )

                                    # Add to our accumulated clarifications
                                    all_clarifications.update(new_clarifications)
                                    debug_print(
                                        f"Updated all_clarifications with new answers"
                                    )

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
                                            "content": json.dumps(new_clarifications),
                                        }
                                    )
                                    debug_print(
                                        "Added tool call and response to message history"
                                    )

                                    # Continue the conversation with a new prompt that includes clarifications
                                    formatted_prompt_with_clarifications = PLANNING_WITH_CLARIFICATION_PROMPT.format(
                                        description=requirements.get(
                                            "notebook_description", "Not provided"
                                        ),
                                        purpose=requirements.get(
                                            "notebook_purpose", "Not provided"
                                        ),
                                        target_audience=requirements.get(
                                            "target_audience", "Not provided"
                                        ),
                                        additional_requirements=format_additional_requirements(
                                            requirements.get(
                                                "additional_requirements", []
                                            )
                                        ),
                                        code_snippets=format_code_snippets(
                                            requirements.get("code_snippets", [])
                                        ),
                                        clarifications=format_clarifications(
                                            all_clarifications
                                        ),
                                    )

                                    # Add this new prompt to continue the conversation
                                    messages.append(
                                        {
                                            "role": "user",
                                            "content": formatted_prompt_with_clarifications,
                                        }
                                    )
                                    debug_print(
                                        "Added clarification-enhanced prompt to message history"
                                    )
                                else:
                                    debug_print(
                                        "No questions found in tool call, ending clarification rounds"
                                    )
                                    continue_clarifying = False
                else:
                    # No more clarification questions, proceed to final plan generation
                    debug_print(
                        "No clarification tool call detected, proceeding to plan generation"
                    )
                    continue_clarifying = False

                    # Use structured output to get the notebook plan
                    debug_print("Generating notebook plan with structured output...")
                    notebook_plan = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        response_format=NotebookPlanModel,
                    )

                    parsed_plan = notebook_plan.choices[0].message.parsed
                    if parsed_plan is not None:
                        debug_print("Successfully parsed notebook plan")
                        return parsed_plan

                    # If parsed result is None, create an error plan
                    debug_print("Failed to parse notebook plan, returning error plan")
                    return self._create_error_plan("Failed to parse notebook plan")

            except Exception as e:
                debug_print(f"Error in plan_notebook: {e}")
                return self._create_error_plan(str(e))

        # Final attempt to generate plan with all accumulated clarifications
        if all_clarifications:
            debug_print(
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
                debug_print("Making final API call for notebook plan...")
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
                    debug_print("Successfully parsed final notebook plan")
                    return parsed_plan

                # If parsed result is None, create an error plan
                debug_print("Failed to parse final notebook plan, returning error plan")
                return self._create_error_plan("Failed to parse notebook plan")

            except Exception as e:
                debug_print(f"Error in final plan generation: {e}")

        # If we reach here, try to generate a plan from the last state
        try:
            debug_print(
                "Attempting fallback plan generation from last message state..."
            )
            notebook_plan = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=NotebookPlanModel,
            )

            parsed_plan = notebook_plan.choices[0].message.parsed
            if parsed_plan is not None:
                debug_print("Successfully parsed fallback notebook plan")
                return parsed_plan

            # If parsed result is None, create an error plan
            debug_print("Failed to parse fallback notebook plan, returning error plan")
            return self._create_error_plan("Failed to parse notebook plan")

        except Exception as e:
            debug_print(f"Error in fallback plan generation: {e}")

        # Return error plan if all attempts fail
        debug_print("All plan generation attempts failed, returning error plan")
        return self._create_error_plan("Failed to generate notebook plan")

    def _create_error_plan(self, error_message: str) -> NotebookPlanModel:
        """Create a basic error plan when notebook generation fails."""
        debug_print(f"Creating error plan: {error_message}")
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
