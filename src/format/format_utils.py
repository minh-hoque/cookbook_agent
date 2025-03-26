"""
Utility functions for formatting data.

This module re-exports functions from the reorganized modules for backward compatibility.
New code should import directly from the specialized modules.

DEPRECATED: Import from specialized modules instead:
- src.format.core_utils: Basic utility functions
- src.format.markdown_utils: Markdown formatting utilities
- src.format.notebook_utils: Notebook conversion utilities
- src.format.prompt_utils: Prompt formatting utilities
- src.format.plan_format: Notebook plan formatting utilities
"""

import warnings

warnings.warn(
    "Importing from format_utils is deprecated. Please import from the specialized modules instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all functions for backward compatibility
from src.format.core_utils import format_json
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
