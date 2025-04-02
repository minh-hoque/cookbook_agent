# Cookbook Agent Test Modules

This directory contains test modules for the different components of the Cookbook Agent project. These tests serve both as functional verification and practical demonstrations of the system.

## Available Tests

- `test_planner.py`: Tests the PlannerLLM implementation
- `test_writer.py`: Tests the WriterAgent implementation (most commonly used)
- `test_searcher.py`: Tests the search functionality
- `test_format.py`: Tests the format utilities

## Running the Tests

### Testing the Writer Agent

The WriterAgent test is the most comprehensive and useful test for most users. It allows you to generate notebook content from a plan.

#### Basic Usage

```bash
# Run with default settings (generates full notebook from default plan)
python -m src.tests.test_writer

# Specify a custom plan file
python -m src.tests.test_writer --plan ./data/agents_sdk_plan.md

# Save output to a specific directory
python -m src.tests.test_writer --output ./my_notebooks
```

#### Command Line Options

The WriterAgent test script accepts the following command line options:

- `--plan`: Path to the markdown plan file (default: `./data/test_notebook_plan.md`)
- `--output`: Directory to save the output to (default: `./output`)
- `--section`: Index of the section to generate content for (0-based, default: None, which means generate all sections)
- `--model`: The model to use for generation (default: `gpt-4o`)
- `--max-retries`: Maximum number of retries for content generation (default: 3)
- `--parse-only`: Only parse the plan and save it as JSON, without generating content
- `--no-search`: Disable search functionality for content generation

#### Testing Specific Sections

To test content generation for just one section:

```bash
# Generate only the first section (index 0)
python -m src.tests.test_writer --section 0

# Generate only the third section (index 2)
python -m src.tests.test_writer --section 2 --plan ./data/complex_plan.md
```

This is useful when:
- You want to quickly iterate on a specific section
- You're testing with a large notebook and don't need the entire content
- You're troubleshooting issues with a particular section

#### Model Selection

You can specify which OpenAI model to use:

```bash
# Use GPT-4o (default)
python -m src.tests.test_writer --model gpt-4o

# Use GPT-3.5 Turbo for faster, less expensive generation
python -m src.tests.test_writer --model gpt-3.5-turbo

# Use GPT-4 Turbo with vision for notebooks that incorporate images
python -m src.tests.test_writer --model gpt-4-turbo
```

#### Error Handling and Retries

Control how the system handles errors during generation:

```bash
# Set maximum number of retries (default is 3)
python -m src.tests.test_writer --max-retries 5

# Disable retries entirely
python -m src.tests.test_writer --max-retries 0
```

Increasing retries is useful when:
- Working with complex topics that might hit model context limits
- Experiencing intermittent API issues
- Using models that might struggle with particular content types

#### Search Integration

Control whether internet search is used during content generation:

```bash
# Disable search (generate content using only the model's knowledge)
python -m src.tests.test_writer --no-search

# Enable search (default behavior)
python -m src.tests.test_writer
```

Disabling search is useful when:
- You want completely reproducible results
- You don't have a Tavily API key configured
- You're testing the model's base knowledge
- You want faster generation (searches add time to the process)

#### Plan Parsing

If you just want to verify your notebook plan without generating content:

```bash
# Parse the plan and save it as JSON (no content generation)
python -m src.tests.test_writer --parse-only

# Useful for checking a custom plan file
python -m src.tests.test_writer --plan ./my_plans/custom_plan.md --parse-only
```

This is useful for:
- Validating the structure of your notebook plan
- Debugging issues with plan parsing
- Quickly checking the parsed representation of your plan

#### Expected Output

When running the writer test, expect:

1. A progress indicator showing which section is being generated
2. Information about search queries (if search is enabled)
3. Notices about saved output files
4. Any errors or warnings during generation

The test will produce the following files (as detailed in the Output Files section):
- Individual JSON files for each section
- Markdown, Jupyter notebook, and Python script versions of the notebook
- When critique is enabled, both original and revised versions plus a critique file

### Testing the Planner

The planner test demonstrates how the system creates a notebook plan from a description:

```bash
# Run the planner test with default settings
python -m src.tests.test_planner

# Specify a custom model
python -m src.tests.test_planner --model gpt-4o
```

When running this test, you'll:
1. Be prompted to enter a notebook description
2. Answer clarification questions (if any)
3. Receive a formatted notebook plan
4. Have the plan saved to the output directory

This test is useful when:
- You want to understand the planning process
- You're developing custom planning prompts
- You want to quickly generate multiple plan outlines

### Testing the Searcher

Test the search integration functionality:

```bash
# Run the basic search test
python -m src.tests.test_searcher

# Test with a specific query
python -m src.tests.test_searcher --query "OpenAI function calling API"

# Test with a specific search provider
python -m src.tests.test_searcher --provider tavily
```

This test demonstrates:
- How search queries are formulated
- How search results are processed
- How the system integrates external information

### Testing Format Utilities

Test the conversion between different content formats:

```bash
# Run the format utility tests
python -m src.tests.test_format

# Test with a specific notebook file
python -m src.tests.test_format --notebook ./output/sample_notebook.ipynb
```

This test shows:
- Conversion between markdown, Jupyter notebooks, and Python scripts
- How notebook content is parsed and processed
- How formatting is preserved across conversions

### Combining Test Parameters

You can combine multiple parameters for customized testing:

```bash
# Generate specific section with a different model and no search
python -m src.tests.test_writer --section 1 --model gpt-3.5-turbo --no-search

# Generate content with custom plan and output directory
python -m src.tests.test_writer --plan ./my_plan.md --output ./custom_output --max-retries 4
```

## Testing Best Practices

1. **Start Simple**: Begin with `--parse-only` to verify your plan structure
2. **Test Single Sections**: Use `--section` to test one section before generating the entire notebook
3. **Iterate on Plans**: Modify your plan file and regenerate to refine content
4. **Compare Models**: Try different models to find the best balance of quality and speed
5. **Version Your Output**: Use different output directories to keep iterations of your notebooks

## Troubleshooting Common Issues

- **API Errors**: Ensure your OpenAI API key is set in your `.env` file
- **Search Errors**: If search fails, try with `--no-search` or ensure your Tavily API key is configured
- **Parse Errors**: If plan parsing fails, check your markdown structure matches the expected format
- **Memory Issues**: For very large notebooks, generate one section at a time with `--section`

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

### Final and Original Files (with Critique)

When critique is enabled (default behavior):
- `<notebook_title>_final.ipynb`: Final Jupyter notebook after critique
- `<notebook_title>_original.ipynb`: Original version before critique
- `<notebook_title>_final.md`: Final markdown version after critique
- `<notebook_title>_original.md`: Original markdown version before critique 
- `<notebook_title>_final.py`: Final Python script after critique
- `<notebook_title>_original.py`: Original Python script before critique
- `<notebook_title>_critique.md`: Critique of the notebook content

## Dependencies

To run the tests, you need to have the following dependencies installed:

```bash
pip install openai langchain-openai langgraph tavily-python
```

## Example Data

The `data/` directory in the project root contains example markdown plans that can be used for testing, including:

- `test_notebook_plan.md`: A simple test plan
- `test_writer_notebook_plan.md`: A plan specifically for testing the WriterAgent
- `agents_sdk_synthetic_transcripts.md`: A more complex plan about OpenAI's Agent APIs