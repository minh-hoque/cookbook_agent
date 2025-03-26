# Format Package

This package provides utilities for formatting and converting generated notebook content into various output formats, including Markdown, Jupyter Notebooks, Python scripts, and JSON.

## Components

The format package contains specialized modules for different formatting operations:

- `core_utils.py`: Basic utility functions for handling data
- `markdown_utils.py`: Converts between notebook content and markdown
- `notebook_utils.py`: Converts between notebook content and Jupyter notebooks/Python scripts
- `plan_format.py`: Handles notebook plan formatting and parsing
- `prompt_utils.py`: Formats data for inclusion in prompts
- `format_utils.py`: Re-exports from specialized modules for backward compatibility

## Key Features

- **Multi-format Export**: Convert notebook content to Markdown, Jupyter, Python, and JSON
- **Format Conversion**: Convert between different formats (Markdown to Notebook and vice versa)
- **Plan Parsing**: Parse notebook plans from markdown files
- **Prompt Formatting**: Format data for inclusion in LLM prompts
- **Pretty Printing**: Format JSON data with proper indentation

## Usage Examples

### Converting WriterAgent Output to Markdown

```python
from src.format.markdown_utils import writer_output_to_markdown
from src.writer import WriterAgent
from src.models import NotebookPlanModel

# Generate content with WriterAgent
writer = WriterAgent()
notebook_content = writer.generate_content(notebook_plan)

# Convert to markdown (returns the markdown string)
markdown = writer_output_to_markdown(notebook_content)

# Or save directly to a file
writer_output_to_markdown(notebook_content, "output/notebook.md")
```

### Converting WriterAgent Output to Jupyter Notebook

```python
from src.format.notebook_utils import writer_output_to_notebook

# Convert to Jupyter notebook
writer_output_to_notebook(
    notebook_content, 
    "output/notebook.ipynb",
    metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    notebook_title="My Awesome Notebook"
)
```

### Converting to Multiple Formats at Once

```python
from src.format.notebook_utils import writer_output_to_files

# Convert to multiple formats
output_files = writer_output_to_files(
    notebook_content,
    output_dir="output",
    notebook_title="MyNotebook",
    formats=["ipynb", "py", "md", "json"]
)

# output_files contains paths to all generated files
print(output_files)
```

### Parsing a Markdown Plan

```python
from src.format.plan_format import parse_markdown_to_plan

# Parse a markdown file into a notebook plan
notebook_plan = parse_markdown_to_plan("data/notebook_plan.md")

# Access plan properties
print(f"Title: {notebook_plan.title}")
print(f"Number of sections: {len(notebook_plan.sections)}")
```

### Converting a Jupyter Notebook to WriterAgent Format

```python
from src.format.notebook_utils import notebook_to_writer_output

# Convert existing notebook to WriterAgent format
notebook_content = notebook_to_writer_output("existing_notebook.ipynb")
```

### Formatting Data for Prompts

```python
from src.format.prompt_utils import format_subsections_details, format_code_snippets

# Format subsections for a prompt
subsections_text = format_subsections_details(notebook_plan.sections[0].subsections)

# Format code snippets for a prompt
code_snippets_text = format_code_snippets(["import openai", "client = OpenAI()"])
```

## Utility Functions By Module

### Core Utils (`core_utils.py`)

- `format_json(data, indent=2)`: Format JSON-serializable data with proper indentation

### Markdown Utils (`markdown_utils.py`)

- `notebook_cell_to_markdown(cell)`: Convert a notebook cell to markdown
- `notebook_section_to_markdown(section)`: Convert a section to markdown
- `notebook_content_to_markdown(sections)`: Convert multiple sections to markdown
- `writer_output_to_markdown(writer_output, output_file=None)`: Convert WriterAgent output to markdown
- `json_file_to_markdown(json_file_path, output_file=None, is_section=True)`: Convert JSON file to markdown
- `markdown_to_notebook_content(markdown_text, section_header_level=2)`: Convert markdown to notebook content

### Notebook Utils (`notebook_utils.py`)

- `writer_output_to_notebook(writer_output, output_file, metadata=None, notebook_title=None)`: Convert to Jupyter notebook
- `writer_output_to_python_script(writer_output, output_file, include_markdown=True)`: Convert to Python script
- `writer_output_to_files(writer_output, output_dir, formats=["ipynb", "py", "md", "json"])`: Convert to multiple formats
- `notebook_to_writer_output(notebook_file, section_header_level=2)`: Convert Jupyter notebook to WriterAgent format
- `save_notebook_versions(original_content, revised_content, critique, output_dir)`: Save original and revised versions

### Plan Format (`plan_format.py`)

- `format_notebook_plan(plan)`: Format notebook plan as markdown
- `save_plan_to_file(plan, output_file)`: Save notebook plan to file
- `parse_markdown_to_plan(markdown_file)`: Parse markdown file into notebook plan

### Prompt Utils (`prompt_utils.py`)

- `format_subsections_details(subsections)`: Format subsections for prompts
- `format_additional_requirements(requirements)`: Format requirements for prompts
- `format_previous_content(previous_content)`: Format previous content for prompts
- `format_code_snippets(snippets)`: Format code snippets for prompts
- `format_clarifications(clarifications)`: Format clarifications for prompts
- `format_cells_for_evaluation(cells)`: Format cells for evaluation

## Batch Conversion

The package includes a utility script for batch converting multiple JSON files to markdown:

```bash
# Convert all JSON files in a directory
python -m src.format.convert_multiple_files --input_dir input_files --output_dir output_files
```

## Requirements

The format package requires the following dependencies:
- Python 3.9+
- Jupyter (for notebook operations)
- Pydantic (for model validation)
- json (standard library)
- os (standard library)
- re (standard library)

These dependencies are included in the project's main requirements.txt file. 