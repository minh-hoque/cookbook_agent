# WriterAgent Test

This directory contains tests for the WriterAgent, which is responsible for generating notebook content based on a plan.

## Running the Tests

### Command Line Interface

You can run the WriterAgent test from the command line using the `run_writer_test.py` script:

```bash
# Run from the project root directory
python src/tests/run_writer_test.py --plan ./data/test_notebook_plan.md --output ./output
```

### Command Line Options

The script accepts the following command line options:

- `--plan`: Path to the markdown plan file (default: `./data/test_notebook_plan.md`)
- `--output`: Directory to save the output to (default: `./output`)
- `--section`: Index of the section to generate content for (0-based, default: None, which means generate all sections)
- `--model`: The model to use for generation (default: `gpt-4o`)
- `--max-retries`: Maximum number of retries for content generation (default: 3)
- `--parse-only`: Only parse the plan and save it as JSON, don't generate content

### Examples

Generate content for all sections using the default plan file:

```bash
python src/tests/run_writer_test.py
```

Generate content for a specific section (e.g., the first section):

```bash
python src/tests/run_writer_test.py --section 0
```

Use a different model:

```bash
python src/tests/run_writer_test.py --model gpt-3.5-turbo
```

Use a custom plan file:

```bash
python src/tests/run_writer_test.py --plan ./path/to/your/plan.md
```

Only parse the plan and save it as JSON (useful for debugging):

```bash
python src/tests/run_writer_test.py --parse-only
```

## Dependencies

To run the full test with content generation, you need to have the following dependencies installed:

```bash
pip install openai langchain-openai langgraph
```

If you don't have these dependencies installed, the test will still parse the plan and save it as JSON, but it won't generate content.

## Markdown Plan Format

The test expects a markdown file with the following format:

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

The `parse_markdown_to_plan` function in `run_writer_test.py` parses this format into a dictionary that can be used to create a `NotebookPlanModel` object for the WriterAgent.

## Output Format

The test saves the generated content in multiple formats in the specified output directory:

1. **JSON Files**: Each section is saved as a separate JSON file with the following structure:

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

3. **Jupyter Notebook (.ipynb)**: The content is saved as a ready-to-use Jupyter notebook file. This notebook can be opened directly in Jupyter Lab, Jupyter Notebook, VS Code, or any other compatible environment.

The output behavior differs slightly depending on whether you're generating a single section or all sections:

### When generating content for a single section

The following files are created:
- A JSON file with the section content (`section_N_SectionTitle.json`)
- A markdown file containing just that section (`notebook_SectionTitle.md`)
- A Jupyter notebook for just that section (`notebook_SectionTitle.ipynb`)

### When generating content for all sections

The following files are created:
- Individual JSON files for each section (`section_1_SectionTitle.json`, `section_2_SectionTitle.json`, etc.)
- A markdown file containing all sections (`notebook_FirstSectionTitle.md` or `full_notebook.md`)
- A complete Jupyter notebook with all sections as a single file (`NotebookTitle.ipynb`)

These files provide flexibility for different use cases and validation methods.

## Error Handling

The test is designed to be robust against import errors and other issues. If there are problems importing the required modules or generating content, the test will still parse the plan and save it as JSON, making it useful for debugging.