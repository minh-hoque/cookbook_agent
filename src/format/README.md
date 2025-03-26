# Format Package

This package provides utilities for formatting various types of data, particularly JSON data from the WriterAgent into readable formats.

## Overview

The format package contains modules for handling different formatting operations:

- `core_utils.py`: Basic utility functions for formatting data.
- `markdown_utils.py`: Functions for converting between markdown and other formats.
- `notebook_utils.py`: Functions for converting between notebook formats and other file types.
- `prompt_utils.py`: Functions for formatting prompt elements.
- `plan_format.py`: Functions for formatting notebook plans.
- `format_utils.py`: (Deprecated) Re-exports from specialized modules for backward compatibility.

## Usage

### Converting WriterAgent Output to Markdown

The main function you'll want to use is `writer_output_to_markdown`, which converts the output of the WriterAgent (a list of `NotebookSectionContent` objects) into a markdown document:

```python
from src.format.markdown_utils import writer_output_to_markdown

# Assuming writer_output is the output from WriterAgent.generate_content()
markdown = writer_output_to_markdown(writer_output)

# You can also save the markdown to a file
writer_output_to_markdown(writer_output, "output.md")
```

### Converting JSON Files to Markdown

If you have JSON files containing notebook content (either a single section or a list of sections), you can convert them to markdown using the `json_file_to_markdown` function:

```python
from src.format.markdown_utils import json_file_to_markdown

# Convert a JSON file containing a single section to markdown
markdown = json_file_to_markdown("path/to/section.json", "output.md", is_section=True)

# Convert a JSON file containing a list of sections to markdown
markdown = json_file_to_markdown("path/to/sections.json", "output.md", is_section=False)
```

### Converting JSON Strings to Markdown

If you have JSON strings containing notebook content, you can convert them to markdown using the `json_string_to_markdown` function:

```python
from src.format.markdown_utils import json_string_to_markdown

# Convert a JSON string containing a single section to markdown
markdown = json_string_to_markdown(json_string, "output.md", is_section=True)

# Convert a JSON string containing a list of sections to markdown
markdown = json_string_to_markdown(json_string, "output.md", is_section=False)
```

### Example

Here's a complete example of how to use the formatting functions:

```python
from src.writer import WriterAgent
from src.models import NotebookPlanModel
from src.format.markdown_utils import writer_output_to_markdown

# Create a writer agent
writer = WriterAgent()

# Assuming you have a notebook_plan
notebook_content = writer.generate_content(notebook_plan)

# Convert to markdown for easy review
markdown = writer_output_to_markdown(notebook_content)

# Save to a file
writer_output_to_markdown(notebook_content, "notebook_output.md")
```

## Available Functions By Module

The package provides the following functions organized by module:

### Core Utils (`core_utils.py`)
- `format_json(data, indent=2)`: Format any JSON-serializable data with proper indentation.

### Markdown Utils (`markdown_utils.py`)
- `notebook_cell_to_markdown(cell)`: Convert a single notebook cell to markdown format.
- `notebook_section_to_markdown(section)`: Convert a notebook section to markdown format.
- `notebook_content_to_markdown(sections)`: Convert a list of notebook sections to a complete markdown document.
- `notebook_plan_to_markdown(plan)`: Convert a notebook plan to markdown format.
- `save_markdown_to_file(markdown, filepath)`: Save markdown content to a file.
- `writer_output_to_markdown(writer_output, output_file=None)`: Convert WriterAgent output to markdown and optionally save to a file.
- `json_file_to_markdown(json_file_path, output_file=None, is_section=True)`: Convert a JSON file containing notebook content to markdown.
- `json_string_to_markdown(json_string, output_file=None, is_section=True)`: Convert a JSON string containing notebook content to markdown.
- `format_notebook_for_critique(notebook_plan, section_contents)`: Format the notebook sections for the final critique.
- `markdown_to_notebook_content(markdown_text, section_header_level=2)`: Convert a markdown string to a list of NotebookSectionContent objects.

### Notebook Utils (`notebook_utils.py`)
- `save_notebook_content(content_list, output_dir)`: Save the generated notebook content to files.
- `writer_output_to_notebook(writer_output, output_file, metadata=None, notebook_title=None)`: Convert WriterAgent output to a Jupyter notebook.
- `writer_output_to_python_script(writer_output, output_file, include_markdown=True)`: Convert WriterAgent output to a Python script.
- `writer_output_to_files(writer_output, output_dir, notebook_title=None, formats=["ipynb", "py", "md"], original_content=None)`: Convert WriterAgent output to multiple file formats.
- `notebook_to_writer_output(notebook_file, section_header_level=2)`: Convert a Jupyter notebook file to WriterAgent output format.
- `save_notebook_versions(original_content, revised_content, critique, output_dir, notebook_title=None, formats=["ipynb", "py", "md"])`: Save original and revised versions of a notebook with critique.

### Prompt Utils (`prompt_utils.py`)
- `format_subsections_details(subsections)`: Format subsections details for prompts.
- `format_additional_requirements(requirements)`: Format additional requirements for prompts.
- `format_previous_content(previous_content)`: Format previously generated content for prompts.
- `format_code_snippets(snippets)`: Format code snippets for prompts.
- `format_clarifications(clarifications)`: Format clarifications for prompts.
- `format_cells_for_evaluation(cells)`: Format cells for evaluation.

### Plan Format (`plan_format.py`)
- `format_notebook_plan(plan)`: Format the notebook plan as a markdown string.
- `save_plan_to_file(plan, output_file)`: Save the formatted notebook plan to a file.
- `parse_markdown_to_plan(markdown_file)`: Parse a markdown file into a NotebookPlanModel.

## Example Scripts

The package includes example scripts that demonstrate how to use the formatting functions. Be sure to update imports to use the specialized modules:

1. `examples.py`: Demonstrates how to convert WriterAgent output to markdown.
   ```bash
   python -m src.format.examples
   ```

2. `json_to_md_example.py`: Demonstrates how to convert JSON files to markdown.
   ```bash
   # Convert a specific JSON file
   python -m src.format.json_to_md_example path/to/file.json [output_file.md]
   
   # Convert the example file (if it exists)
   python -m src.format.json_to_md_example
   ```

These scripts will generate markdown files based on the input JSON data. 