"""
Pydantic models for structured outputs.

This module contains the Pydantic models used for structured outputs from the LLMs.
"""

from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field


class ClarificationQuestions(BaseModel):
    """Model for clarification questions."""

    questions: List[str]
    reasoning: str


class SubSubSection(BaseModel):
    """Model for a sub-subsection in the notebook plan."""

    title: str
    description: str


class SubSection(BaseModel):
    """Model for a subsection in the notebook plan."""

    title: str
    description: str
    subsections: Optional[List[SubSubSection]] = None


class Section(BaseModel):
    """Model for a section in the notebook plan."""

    title: str
    description: str
    subsections: Optional[List[SubSection]] = None


class NotebookPlanModel(BaseModel):
    """Model for the complete notebook plan."""

    title: str
    description: str
    purpose: str
    target_audience: str
    sections: List[Section]


class SearchQuery(BaseModel):
    """Model for search queries."""

    search_query: str = Field(description="Query that is optimized for web search.")
    justification: str = Field(
        description="Why this query is relevant to the user's request."
    )


class NotebookCell(BaseModel):
    """Model for a notebook cell."""

    cell_type: Literal["markdown", "code"] = Field(
        description="Type of the cell (markdown or code)"
    )
    content: str = Field(description="Content of the cell")


class NotebookSectionContent(BaseModel):
    """Model for the content of a notebook section."""

    section_title: str = Field(description="Title of the section")
    cells: List[NotebookCell] = Field(description="List of cells in the section")


class CriticEvaluation(BaseModel):
    """Model for the critic evaluation of notebook content.

    This model matches the expected output format in the critic prompt.
    """

    rationale: str = Field(
        description="Clear and specific reasons supporting the pass or fail decision following the evaluation criteria, including actionable suggestions for improvement if necessary."
    )
    passed: bool = Field(
        description="Whether the content is acceptable as is or requires revision. True if the content is acceptable as is, false if it requires revision.",
    )
