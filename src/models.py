"""
Pydantic models for structured outputs.

This module contains the Pydantic models used for structured outputs from the LLMs.
"""

from typing import List, Optional
from pydantic import BaseModel


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
