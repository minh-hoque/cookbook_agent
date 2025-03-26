"""
Format package for handling various formatting operations.

This package contains modules for formatting different types of data,
particularly JSON data from the WriterAgent into readable formats.
"""

from src.format.core_utils import (
    format_json,
)

from src.format.markdown_utils import (
    notebook_cell_to_markdown,
    notebook_section_to_markdown,
    notebook_content_to_markdown,
    notebook_plan_to_markdown,
    save_markdown_to_file,
    writer_output_to_markdown,
    json_file_to_markdown,
    json_string_to_markdown,
    format_notebook_for_critique,
    markdown_to_notebook_content,
)

from src.format.notebook_utils import (
    save_notebook_content,
    writer_output_to_notebook,
    writer_output_to_python_script,
    writer_output_to_files,
    notebook_to_writer_output,
    save_notebook_versions,
)

from src.format.prompt_utils import (
    format_subsections_details,
    format_additional_requirements,
    format_previous_content,
    format_code_snippets,
    format_clarifications,
    format_cells_for_evaluation,
)

from src.format.plan_format import (
    format_notebook_plan,
    save_plan_to_file,
    parse_markdown_to_plan,
)

__all__ = [
    # Core utilities
    "format_json",
    # Markdown utilities
    "notebook_cell_to_markdown",
    "notebook_section_to_markdown",
    "notebook_content_to_markdown",
    "notebook_plan_to_markdown",
    "save_markdown_to_file",
    "writer_output_to_markdown",
    "json_file_to_markdown",
    "json_string_to_markdown",
    "format_notebook_for_critique",
    "markdown_to_notebook_content",
    # Notebook utilities
    "save_notebook_content",
    "writer_output_to_notebook",
    "writer_output_to_python_script",
    "writer_output_to_files",
    "notebook_to_writer_output",
    "save_notebook_versions",
    # Prompt utilities
    "format_subsections_details",
    "format_additional_requirements",
    "format_previous_content",
    "format_code_snippets",
    "format_clarifications",
    "format_cells_for_evaluation",
    # Plan format utilities
    "format_notebook_plan",
    "save_plan_to_file",
    "parse_markdown_to_plan",
]
