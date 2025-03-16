# Format Package

This package provides utilities for formatting various types of data, particularly JSON data from the WriterAgent into readable formats.

## Overview

The format package contains modules for handling different formatting operations:

- `format_utils.py`: Functions for formatting data, particularly for converting WriterAgent output into markdown for easy review.

## Usage

### Converting WriterAgent Output to Markdown

The main function you'll want to use is `writer_output_to_markdown`, which converts the output of the WriterAgent (a list of `NotebookSectionContent` objects) into a markdown document:

```python
from src.format import writer_output_to_markdown

# Assuming writer_output is the output from WriterAgent.generate_content()
markdown = writer_output_to_markdown(writer_output)

# You can also save the markdown to a file
writer_output_to_markdown(writer_output, "output.md")
```

### Converting JSON Files to Markdown

If you have JSON files containing notebook content (either a single section or a list of sections), you can convert them to markdown using the `json_file_to_markdown` function:

```python
from src.format import json_file_to_markdown

# Convert a JSON file containing a single section to markdown
markdown = json_file_to_markdown("path/to/section.json", "output.md", is_section=True)

# Convert a JSON file containing a list of sections to markdown
markdown = json_file_to_markdown("path/to/sections.json", "output.md", is_section=False)
```

### Converting JSON Strings to Markdown

If you have JSON strings containing notebook content, you can convert them to markdown using the `json_string_to_markdown` function:

```python
from src.format import json_string_to_markdown

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
from src.format import writer_output_to_markdown

# Create a writer agent
writer = WriterAgent()

# Assuming you have a notebook_plan
notebook_content = writer.generate_content(notebook_plan)

# Convert to markdown for easy review
markdown = writer_output_to_markdown(notebook_content)

# Save to a file
writer_output_to_markdown(notebook_content, "notebook_output.md")
```

## Available Functions

The package provides the following functions:

- `format_json(data, indent=2)`: Format any JSON-serializable data with proper indentation.
- `notebook_cell_to_markdown(cell)`: Convert a single notebook cell to markdown format.
- `notebook_section_to_markdown(section)`: Convert a notebook section to markdown format.
- `notebook_content_to_markdown(sections)`: Convert a list of notebook sections to a complete markdown document.
- `notebook_plan_to_markdown(plan)`: Convert a notebook plan to markdown format.
- `save_markdown_to_file(markdown, filepath)`: Save markdown content to a file.
- `writer_output_to_markdown(writer_output, output_file=None)`: Convert WriterAgent output to markdown and optionally save to a file.
- `json_file_to_markdown(json_file_path, output_file=None, is_section=True)`: Convert a JSON file containing notebook content to markdown.
- `json_string_to_markdown(json_string, output_file=None, is_section=True)`: Convert a JSON string containing notebook content to markdown.

## Example Scripts

The package includes example scripts that demonstrate how to use the formatting functions:

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