# Cookbook Agent - PlannerLLM

This implementation provides a PlannerLLM class for generating structured educational notebook plans about OpenAI APIs. The PlannerLLM uses a tool-based approach to dynamically ask for clarifications from users when needed, and leverages OpenAI's structured outputs feature for type-safe return values.

## Features

- **Requirements Analysis**: Analyzes user requirements to generate a notebook plan
- **Interactive Clarification**: Uses tools to ask clarification questions when information is ambiguous or missing
- **Structured Outputs**: Utilizes OpenAI's beta.chat.completions.parse for type-safe structured outputs
- **Robust Error Handling**: Provides fallbacks and graceful error reporting
- **Modular Tools**: Organized reusable tools in a dedicated `tools/` package

## Project Structure

```
src/
├── models.py               # Pydantic models for structured output
├── planner.py              # Main PlannerLLM implementation
├── prompts/                # Prompt templates
│   └── planner_prompts.py  # Templates for planning and clarification
├── tools/                  # LLM tools
│   ├── __init__.py         # Tool exports
│   ├── clarification_tools.py  # Tools for getting clarifications
│   └── utils.py            # Tool utility functions
├── setup_tools.py          # Script to set up tools in Python path
└── test_planner.py         # Test script
```

## Implementation Details

The core implementation consists of:

1. **Pydantic Models**: Strong type definitions for both clarification questions and notebook plans
2. **Prompt Templates**: Templates for generating the initial plan and plans with clarifications
3. **Tool-based Interaction**: Uses OpenAI function calling to request clarifications from users
4. **Structured Outputs**: Utilizes OpenAI's structured outputs to ensure type-safe responses
5. **Tools Package**: Reusable tool definitions with utility functions for working with LLM tools

## Using the Tools Package

The tools package can be imported in two ways:

### Method 1: Add to Python Path (Recommended)

Run the `setup_tools.py` script to add the tools package to your Python path:

```bash
# From the project root
python src/setup_tools.py
```

After running the setup script, you can import tools directly:

```python
from tools import get_clarifications_tool, extract_tool_arguments, has_tool_call
```

### Method 2: Relative Imports

```python
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now you can import from tools
from tools import get_clarifications_tool
```

## Usage Example

```python
from planner import PlannerLLM

# Create a callback to handle clarification questions
def callback(questions):
    # In a real application, this would prompt the user and collect answers
    answers = {}
    for question in questions:
        answer = input(f"Question: {question}\nAnswer: ")
        answers[question] = answer
    return answers

# Initialize the planner
planner = PlannerLLM()

# Generate a notebook plan
notebook_plan = planner.plan_notebook(
    {
        "notebook_description": "Guide to OpenAI APIs",
        "notebook_purpose": "Educational tutorial",
        "target_audience": "Beginners",
        "additional_requirements": ["Include code examples", "Cover authentication"],
        "code_snippets": ["import openai\nclient = OpenAI()\n..."]
    },
    callback
)

# Access the structured plan
print(f"Notebook Title: {notebook_plan.title}")
print(f"Sections: {len(notebook_plan.sections)}")
```

## Testing

You can run the test script to see the PlannerLLM in action:

```bash
cd src
python test_planner.py
``` 