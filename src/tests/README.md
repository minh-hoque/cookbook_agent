# Cookbook Agent Test Modules

This directory contains test modules for the different components of the Cookbook Agent project.

## Available Tests

- `test_planner.py`: Tests the PlannerLLM implementation
- `test_writer.py`: Tests the WriterAgent implementation
- `test_searcher.py`: Tests the search functionality
- `test_format.py`: Tests the format utilities

## Running the Tests

### Testing the Writer Agent

The WriterAgent test is the most comprehensive and can be run from the command line:

```bash
# Run from the project root directory
python -m src.tests.test_writer --plan ./data/test_notebook_plan.md --output ./output
```

### Command Line Options

The WriterAgent test script accepts the following command line options:

- `--plan`: Path to the markdown plan file (default: `./data/test_notebook_plan.md`)
- `--output`: Directory to save the output to (default: `./output`)
- `--section`: Index of the section to generate content for (0-based, default: None, which means generate all sections)
- `--model`: The model to use for generation (default: `gpt-4o`)
- `--max-retries`: Maximum number of retries for content generation (default: 3)
- `--parse-only`: Only parse the plan and save it as JSON, without generating content
- `--no-search`: Disable search functionality for content generation

### Examples

Generate content for all sections using the default plan file:

```bash
python -m src.tests.test_writer
```

Generate content for a specific section (e.g., the first section):

```bash
python -m src.tests.test_writer --section 0
```

Use a different model:

```bash
python -m src.tests.test_writer --model gpt-3.5-turbo
```

Use a custom plan file:

```bash
python -m src.tests.test_writer --plan ./path/to/your/plan.md
```

Only parse the plan and save it as JSON:

```bash
python -m src.tests.test_writer --parse-only
```

### Testing the Planner

The PlannerLLM test demonstrates the planning functionality:

```bash
python -m src.tests.test_planner
```

### Testing the Searcher

The searcher test demonstrates the search functionality:

```bash
python -m src.tests.test_searcher
```

### Testing the Format Utilities

The format utilities test demonstrates the conversion between different formats:

```bash
python -m src.tests.test_format
```

## Dependencies

To run the tests, you need to have the following dependencies installed:

```bash
pip install openai langchain-openai langgraph tavily-python
```

## Markdown Plan Format

The writer test expects a markdown file with the following format:

```markdown
# Title of the Notebook

**Description:** A brief description of the notebook.

**Purpose:** The purpose of the notebook.

**Target Audience:** The target audience for the notebook.

## Outline

### Section 1 Title

Section 1 description.

#### Subsection 1.1 Title

Subsection 1.1 description.

##### Sub-subsection 1.1.1 Title

Sub-subsection 1.1.1 description.

### Section 2 Title

Section 2 description.
```

## Output Formats

The WriterAgent test saves the generated content in multiple formats:

1. **JSON Files**: Each section is saved as a separate JSON file:

```json
{
  "section_title": "Section Title",
  "cells": [
    {
      "cell_type": "markdown",
      "content": "Markdown content"
    },
    {
      "cell_type": "code",
      "content": "Code content"
    }
  ]
}
```

2. **Markdown File**: A complete markdown version of the notebook is saved as a single file.

3. **Jupyter Notebook (.ipynb)**: The content is saved as a ready-to-use Jupyter notebook file.

4. **Python Script (.py)**: The content is saved as a Python script with code cells and comments.

### Output Files When Generating a Single Section

- A JSON file with the section content (`section_N_SectionTitle.json`)
- A markdown file containing just that section (`notebook_SectionTitle.md`)
- A Jupyter notebook for just that section (`notebook_SectionTitle.ipynb`)
- A Python script for just that section (`notebook_SectionTitle.py`)

### Output Files When Generating All Sections

- Individual JSON files for each section (`section_1_SectionTitle.json`, `section_2_SectionTitle.json`, etc.)
- A markdown file containing all sections (`notebook_NotebookTitle.md`)
- A complete Jupyter notebook with all sections (`notebook_NotebookTitle.ipynb`)
- A Python script with all sections (`notebook_NotebookTitle.py`)

## Error Handling

The tests are designed to be robust against import errors and other issues. If there are problems importing required modules or generating content, the tests will provide descriptive error messages and fallback to simpler operations where possible.

## Example Data

The `data/` directory in the project root contains example markdown plans that can be used for testing, including:

- `test_notebook_plan.md`: A simple test plan
- `test_writer_notebook_plan.md`: A plan specifically for testing the WriterAgent
- `agents_sdk_synthetic_transcripts.md`: A more complex plan about OpenAI's Agent APIs