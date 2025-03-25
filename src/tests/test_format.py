#!/usr/bin/env python3
"""
Test the format module functionality.

This module contains tests for the format module, specifically
testing the functions that handle plan formatting and parsing.
"""

import os
import unittest
import tempfile
from src.models import NotebookPlanModel, Section, SubSection, SubSubSection
from src.format.plan_format import (
    format_notebook_plan,
    save_plan_to_file,
    parse_markdown_to_plan,
)


class TestPlanFormatting(unittest.TestCase):
    """Test the plan_format module functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample notebook plan for testing
        self.sample_plan = NotebookPlanModel(
            title="Test Notebook",
            description="A test notebook for unit testing",
            purpose="To test the format module",
            target_audience="Developers",
            sections=[
                Section(
                    title="Section 1",
                    description="This is the first section",
                    subsections=[
                        SubSection(
                            title="Subsection 1.1",
                            description="This is subsection 1.1",
                            subsections=[
                                SubSubSection(
                                    title="Sub-subsection 1.1.1",
                                    description="This is sub-subsection 1.1.1",
                                )
                            ],
                        )
                    ],
                ),
                Section(
                    title="Section 2",
                    description="This is the second section",
                    subsections=None,
                ),
            ],
        )

    def test_format_notebook_plan(self):
        """Test the format_notebook_plan function."""
        formatted_plan = format_notebook_plan(self.sample_plan)

        # Check that the formatted plan contains the expected elements
        self.assertIn("# Test Notebook", formatted_plan)
        self.assertIn(
            "**Description:** A test notebook for unit testing", formatted_plan
        )
        self.assertIn("**Purpose:** To test the format module", formatted_plan)
        self.assertIn("**Target Audience:** Developers", formatted_plan)
        self.assertIn("### 1. Section 1", formatted_plan)
        self.assertIn("This is the first section", formatted_plan)
        self.assertIn("#### 1.1. Subsection 1.1", formatted_plan)
        self.assertIn("This is subsection 1.1", formatted_plan)
        self.assertIn("##### 1.1.1. Sub-subsection 1.1.1", formatted_plan)
        self.assertIn("This is sub-subsection 1.1.1", formatted_plan)
        self.assertIn("### 2. Section 2", formatted_plan)
        self.assertIn("This is the second section", formatted_plan)

    def test_save_and_parse_plan(self):
        """Test the save_plan_to_file and parse_markdown_to_plan functions together."""
        # Create a temporary file to save the plan to
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as temp_file:
            temp_filename = temp_file.name

        try:
            # Save the plan to the temporary file
            save_plan_to_file(self.sample_plan, temp_filename)

            # Parse the plan back from the file
            parsed_plan = parse_markdown_to_plan(temp_filename)

            print(f"Parsed plan: {parsed_plan}")

            # Check that the parsed plan matches the original
            self.assertEqual(parsed_plan.title, self.sample_plan.title)
            self.assertEqual(parsed_plan.description, self.sample_plan.description)
            self.assertEqual(parsed_plan.purpose, self.sample_plan.purpose)
            self.assertEqual(
                parsed_plan.target_audience, self.sample_plan.target_audience
            )

            # Check sections
            self.assertEqual(len(parsed_plan.sections), len(self.sample_plan.sections))

            # Check first section
            self.assertEqual(
                parsed_plan.sections[0].title, self.sample_plan.sections[0].title
            )
            self.assertIn(
                self.sample_plan.sections[0].description.strip(),
                parsed_plan.sections[0].description.strip(),
            )

            # Check subsections of first section
            # Make sure subsections exist before checking
            self.assertTrue(parsed_plan.sections[0].subsections is not None)
            self.assertTrue(self.sample_plan.sections[0].subsections is not None)

            if (
                parsed_plan.sections[0].subsections
                and self.sample_plan.sections[0].subsections
            ):
                self.assertEqual(
                    len(parsed_plan.sections[0].subsections),
                    len(self.sample_plan.sections[0].subsections),
                )
                self.assertEqual(
                    parsed_plan.sections[0].subsections[0].title,
                    self.sample_plan.sections[0].subsections[0].title,
                )

            # Check second section
            self.assertEqual(
                parsed_plan.sections[1].title, self.sample_plan.sections[1].title
            )
            self.assertIn(
                self.sample_plan.sections[1].description.strip(),
                parsed_plan.sections[1].description.strip(),
            )

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    def test_parse_markdown_direct(self):
        """Test parse_markdown_to_plan with a direct markdown string."""
        # Create a temporary markdown file with known content
        markdown_content = """# Direct Test Notebook

**Description:** A direct test notebook
**Purpose:** To test markdown parsing
**Target Audience:** Test users

## Outline

### 1. First Section

This is the first section description.

#### 1.1. First Subsection

This is the first subsection description.

##### 1.1.1. First Sub-subsection

This is the first sub-subsection description.

### 2. Second Section

This is the second section description.
"""
        with tempfile.NamedTemporaryFile(
            suffix=".md", mode="w", delete=False
        ) as temp_file:
            temp_file.write(markdown_content)
            temp_filename = temp_file.name

        try:
            # Parse the markdown file
            parsed_plan = parse_markdown_to_plan(temp_filename)

            # Verify the parsed content
            self.assertEqual(parsed_plan.title, "Direct Test Notebook")
            self.assertEqual(parsed_plan.description, "A direct test notebook")
            self.assertEqual(parsed_plan.purpose, "To test markdown parsing")
            self.assertEqual(parsed_plan.target_audience, "Test users")

            # Verify sections
            self.assertEqual(len(parsed_plan.sections), 2)
            self.assertEqual(parsed_plan.sections[0].title, "First Section")
            self.assertIn(
                "This is the first section description",
                parsed_plan.sections[0].description,
            )

            # Verify subsections
            if parsed_plan.sections[0].subsections:
                self.assertEqual(len(parsed_plan.sections[0].subsections), 1)
                self.assertEqual(
                    parsed_plan.sections[0].subsections[0].title, "First Subsection"
                )

                # Verify sub-subsections
                if parsed_plan.sections[0].subsections[0].subsections:
                    self.assertEqual(
                        len(parsed_plan.sections[0].subsections[0].subsections), 1
                    )
                    self.assertEqual(
                        parsed_plan.sections[0].subsections[0].subsections[0].title,
                        "First Sub-subsection",
                    )

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)


if __name__ == "__main__":
    unittest.main()
