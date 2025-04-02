# Cookbook Agent - Core Modules

This directory contains the core implementation modules for the Cookbook Agent project, which generates high-quality educational notebook content about OpenAI APIs and AI development concepts.

## Main Components

- **PlannerLLM**: Analyzes user requirements and generates structured notebook outlines with clarification capabilities
- **WriterAgent**: Generates comprehensive notebook content using a sophisticated LangGraph workflow
- **Searcher**: Integrates web search capabilities (OpenAI and Tavily) to enhance content with up-to-date information
- **Critic**: Provides feedback on generated content for quality improvement
- **Format Utilities**: Exports content in multiple formats (Markdown, Jupyter, Python scripts)

## Architecture Overview

The system follows a sophisticated pipeline with feedback loops:

1. **Planning Stage**: The `PlannerLLM` analyzes requirements, asks clarifying questions, and generates a detailed outline
2. **Writing Stage**: The `WriterAgent` generates content following a multi-step workflow:
   - Content generation for each section
   - Dynamic search integration when additional information is needed
   - Intermediate critique and revision
3. **Critique Stage**: The generated content undergoes a comprehensive review
   - Accuracy and correctness evaluation
   - Code quality assessment
   - Content completeness verification
4. **Revision Stage**: Content is refined based on critique feedback
5. **Export Stage**: The format utilities convert the content to various output formats



## Key Features

- **Interactive Clarification**: Advanced tools to ask clarification questions when requirements are ambiguous or incomplete
- **Structured Outputs**: Type-safe models using Pydantic for reliable data handling
- **Search Integration**: Both Tavily and OpenAI search capabilities with configurable context sizes
- **Graph-based Workflow**: LangGraph for orchestrating complex agent interactions with state management
- **Auto-critique**: Comprehensive content quality evaluation and revision loop
- **Multi-format Export**: Generation of different file formats with consistent styling and formatting
- **Error Recovery**: Robust error handling with automatic retries and fallback mechanisms

## PlannerLLM in Depth

The `PlannerLLM` is responsible for creating well-structured notebook plans:

### Key Capabilities

- **Requirement Analysis**: Parses and interprets user requirements
- **Clarification Generation**: Identifies ambiguities and generates targeted questions
- **Structured Planning**: Creates hierarchical notebook outlines with sections and subsections
- **Audience Adaptation**: Adjusts content depth and complexity based on target audience
- **Search Enhancement**: Integrates search results to improve plan quality

### Advanced Usage

```python
from src.planner import PlannerLLM
from src.models import NotebookPlanModel
from src.searcher import search_with_openai, format_openai_search_results

# Create a callback to handle clarification questions
def get_clarifications(questions):
    answers = {}
    for question in questions:
        print(f"Question: {question}")
        answer = input("Answer: ")
        answers[question] = answer
    return answers

# Get search results to enhance planning
search_results = search_with_openai("OpenAI Assistant API capabilities")
formatted_search = format_openai_search_results(search_results)

# Initialize the planner with custom parameters
planner = PlannerLLM(
    model="gpt-4o",
    max_clarification_rounds=2,
    temperature=0.7
)

# Generate a notebook plan with search enhancement
notebook_plan = planner.plan_notebook(
    {
        "notebook_description": "Guide to using OpenAI Assistant API",
        "notebook_purpose": "Educational tutorial",
        "target_audience": "AI developers",
        "additional_requirements": [
            "Include real-world examples", 
            "Cover error handling",
            "Discuss performance optimization"
        ],
    },
    clarification_callback=get_clarifications,
    search_results=formatted_search
)

# Save the plan to markdown
from src.format.plan_format import save_plan_to_file
save_plan_to_file(notebook_plan, "assistant_api_plan.md")
```

## WriterAgent in Depth

The `WriterAgent` implements a sophisticated LangGraph workflow for high-quality content generation:

### Key Components

- **Section Generator**: Produces detailed content for each section of the notebook
- **Search Decision Node**: Determines when additional information is needed
- **Searcher Integration**: Retrieves relevant information from the web
- **Critic**: Evaluates content quality, accuracy, and completeness
- **Revision Node**: Refines content based on critique feedback

### Advanced Configuration

```python
from src.writer import WriterAgent
from src.models import NotebookPlanModel

# Load an existing plan
from src.format.plan_format import parse_markdown_to_plan
notebook_plan = parse_markdown_to_plan("./data/assistant_api_plan.md")

# Configure a writer agent with advanced settings
writer = WriterAgent(
    model="gpt-4o",
    search_enabled=True,
    search_provider="openai",  # or "tavily"
    search_context_size="high",  # "low", "medium", or "high"
    max_retries=5,
    final_critique_enabled=True,
    temperature=0.7
)

# Generate content for specific sections only
section_indices = [0, 2]  # Generate only sections 1 and 3
section_contents = writer.generate_specific_sections(
    notebook_plan=notebook_plan,
    section_indices=section_indices
)

# Generate content for the entire notebook with critique and revision
original_content, critique, revised_content = writer.generate_content(
    notebook_plan=notebook_plan,
    additional_requirements=["Ensure all code examples are runnable"]
)

# Save content with multiple formats
from src.format.notebook_utils import save_notebook_versions
output_files = save_notebook_versions(
    original_content=original_content,
    revised_content=revised_content,
    critique=critique,
    output_dir="./output",
    notebook_title=notebook_plan.title,
    formats=["ipynb", "md", "py"]
)

print(f"Generated files: {output_files}")
```

## Searcher in Depth

The `searcher` module provides sophisticated search capabilities to enhance content with up-to-date information:

### Search Providers

- **OpenAI Search**: Leverages OpenAI's web search capabilities
- **Tavily Search**: Uses the Tavily API for specialized search results

### Search Context Sizes

- **Low**: Basic search with minimal context (faster, less token usage)
- **Medium**: Balanced search with moderate context
- **High**: Comprehensive search with detailed context (slower, more token usage)

### Advanced Search Usage

```python
from src.searcher import (
    search_with_tavily, 
    search_with_openai, 
    format_tavily_search_results,
    format_openai_search_results
)

# Advanced Tavily search with filtering
tavily_results = search_with_tavily(
    query="OpenAI Assistant API capabilities",
    max_results=8,
    search_depth="comprehensive",
    include_domains=["openai.com", "platform.openai.com"],
    exclude_domains=["reddit.com", "medium.com"]
)
formatted_tavily = format_tavily_search_results(tavily_results)

# OpenAI search with different context sizes
openai_basic = search_with_openai(
    query="OpenAI function calling API",
    search_context_size="low"
)
openai_detailed = search_with_openai(
    query="OpenAI function calling API",
    search_context_size="high",
    model="gpt-4-turbo"
)

# Combined search strategy
def combined_search(query):
    # Try OpenAI search first
    try:
        results = search_with_openai(query)
        if results and not results.get("error"):
            return format_openai_search_results(results)
    except Exception as e:
        print(f"OpenAI search failed: {e}")
    
    # Fall back to Tavily search
    try:
        results = search_with_tavily(query)
        return format_tavily_search_results(results)
    except Exception as e:
        print(f"Tavily search failed: {e}")
        return "Search unavailable. Using model knowledge only."

# Use the combined search
search_results = combined_search("Latest OpenAI model capabilities")
```

## Critique and Revision Process

The critique and revision process is a key feature that improves content quality:

### Critique Evaluation Criteria

- **Factual Accuracy**: Ensuring information is correct and up-to-date
- **Code Quality**: Checking that code examples are functional and follow best practices
- **Content Completeness**: Verifying that all required topics are covered
- **Educational Value**: Assessing how well the content teaches the concepts
- **Structure and Flow**: Evaluating the logical progression of information

### Revision Strategies

- **Fact Correction**: Updating incorrect or outdated information
- **Code Improvement**: Enhancing code examples for clarity and functionality
- **Content Expansion**: Adding missing information or examples
- **Flow Enhancement**: Reorganizing content for better educational flow



## Format Utilities

The `format` package provides comprehensive utilities for converting and exporting content:

### Supported Formats

- **Markdown**: Clean, readable documentation format
- **Jupyter Notebook**: Interactive notebook with code and markdown cells
- **Python Script**: Executable Python file with code and comments
- **JSON**: Structured data format for programmatic access

### Advanced Format Operations

```python
from src.format.notebook_utils import (
    writer_output_to_notebook,
    writer_output_to_python_script,
    notebook_to_writer_output,
    save_notebook_versions
)
from src.format.markdown_utils import writer_output_to_markdown
from src.format.plan_format import format_notebook_plan, parse_markdown_to_plan

# Convert existing Jupyter notebook to writer output format
imported_content = notebook_to_writer_output(
    notebook_file="./existing_notebook.ipynb",
    section_header_level=2  # Define what header level constitutes a section
)

# Add custom metadata to exported notebook
notebook_metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.9.0"
    },
    "authors": ["Cookbook Agent"],
    "creation_date": "2023-05-10"
}

# Create Jupyter notebook with custom metadata
jupyter_output = writer_output_to_notebook(
    writer_output=generated_content,
    output_file="./output/advanced_notebook.ipynb",
    metadata=notebook_metadata,
    notebook_title="Advanced OpenAI API Guide",
    return_notebook=True  # Return the notebook dict instead of a boolean
)

# Convert multiple formats at once
from src.format.notebook_utils import writer_output_to_files
output_files = writer_output_to_files(
    writer_output=revised_content,
    output_dir="./output/multi_format",
    notebook_title="OpenAI API Guide",
    formats=["ipynb", "py", "md"],
    original_content=original_content  # Optional: include both versions
)
```