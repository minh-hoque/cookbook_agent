# Cookbook Agent - Core Modules

This directory contains the core implementation modules for the Cookbook Agent project, which generates high-quality educational notebook content about OpenAI APIs and AI development concepts.

## Main Components

- **PlannerLLM**: Analyzes user requirements and generates structured notebook outlines
- **WriterAgent**: Generates comprehensive notebook content using a LangGraph workflow
- **Searcher**: Integrates web search capabilities to enhance content with up-to-date information
- **Format Utilities**: Exports content in multiple formats (Markdown, Jupyter, Python scripts)

## Architecture Overview

The system follows a three-stage pipeline:

1. **Planning Stage**: The `PlannerLLM` analyzes requirements and generates a detailed outline
2. **Writing Stage**: The `WriterAgent` generates content following a critiquing and revision workflow
3. **Export Stage**: The format utilities convert the content to various output formats

## Key Features

- **Interactive Clarification**: Tools to ask clarification questions when requirements are ambiguous
- **Structured Outputs**: Type-safe models using Pydantic
- **Search Integration**: Tavily and OpenAI search capabilities for up-to-date information
- **Graph-based Workflow**: LangGraph for orchestrating agent interactions with state management
- **Auto-critique**: Content quality evaluation and revision loop

## Module Structure

```
src/
├── models.py               # Pydantic models for structured output
├── planner.py              # PlannerLLM implementation
├── writer.py               # WriterAgent implementation
├── searcher.py             # Search functionality
├── user_input.py           # User input utilities
├── format/                 # Formatting utilities
│   ├── core_utils.py       # Basic utility functions
│   ├── markdown_utils.py   # Markdown formatting
│   ├── notebook_utils.py   # Notebook conversion
│   ├── prompt_utils.py     # Prompt formatting utilities
│   └── plan_format.py      # Plan formatting utilities
├── prompts/                # Prompt templates
│   ├── planner_prompts.py  # Templates for planning
│   ├── writer_prompts.py   # Templates for content writing
│   ├── critic_prompts.py   # Templates for content evaluation
│   └── search_prompts.py   # Templates for search operations
├── tools/                  # LLM tools
│   ├── clarification_tools.py # Tools for getting clarifications
│   ├── debug.py            # Debug utilities for tools
│   └── utils.py            # Tool utility functions
├── utils/                  # General utilities
└── tests/                  # Test modules
```

## Using PlannerLLM

The `PlannerLLM` class analyzes user requirements, asks for clarifications when needed, and generates a structured notebook plan:

```python
from src.planner import PlannerLLM
from src.models import NotebookPlanModel

# Create a callback to handle clarification questions
def get_clarifications(questions):
    answers = {}
    for question in questions:
        print(f"Question: {question}")
        answer = input("Answer: ")
        answers[question] = answer
    return answers

# Initialize the planner
planner = PlannerLLM(model="gpt-4o")

# Generate a notebook plan
notebook_plan = planner.plan_notebook(
    {
        "notebook_description": "Guide to using OpenAI Assistant API",
        "notebook_purpose": "Educational tutorial",
        "target_audience": "AI developers",
        "additional_requirements": ["Include real-world examples", "Cover error handling"],
    },
    clarification_callback=get_clarifications
)
```

## Using WriterAgent

The `WriterAgent` uses a LangGraph workflow to generate high-quality content based on the notebook plan:

```python
from src.writer import WriterAgent

# Initialize the writer agent
writer = WriterAgent(model="gpt-4o", search_enabled=True)

# Generate content for the entire notebook
notebook_content = writer.generate_content(notebook_plan)

# Save the generated content
writer.save_to_directory(
    notebook_plan=notebook_plan,
    section_contents=notebook_content,
    output_dir="./output",
    formats=["ipynb", "md", "py", "json"]
)
```

## Using Searcher

The `searcher` module provides tools to search for information about notebook topics:

```python
from src.searcher import search_with_tavily, search_with_openai, format_tavily_search_results

# Search with Tavily
search_results = search_with_tavily("OpenAI Assistant API capabilities", max_results=5)
formatted_results = format_tavily_search_results(search_results)

# Search with OpenAI
openai_results = search_with_openai("OpenAI Assistant API capabilities", model="gpt-4o")
```

## Format Utilities

The `format` package provides utilities for converting notebook content to different formats:

```python
from src.format.markdown_utils import writer_output_to_markdown
from src.format.notebook_utils import writer_output_to_notebook

# Convert to Markdown
markdown = writer_output_to_markdown(notebook_content, output_file="notebook.md")

# Convert to Jupyter Notebook
writer_output_to_notebook(notebook_content, output_file="notebook.ipynb")
```

## Installation

This package is designed to be installed via:

```bash
# From project root
pip install -e .
```

## Requirements

The core requirements include:
- Python 3.9+
- OpenAI Python SDK
- LangChain and LangGraph
- Pydantic
- Tavily Python SDK (optional, for search functionality)
- Jupyter (for notebook handling)

See the project's `requirements.txt` for full dependencies. 