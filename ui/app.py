#!/usr/bin/env python3
"""
Streamlit UI for the OpenAI Demo Notebook Generator
"""

import streamlit as st
import sys
import os
from pathlib import Path
import json
from typing import List, Sequence, Union, cast, Dict, Optional
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import streamlit.components.v1 as components
from queue import Queue
from threading import Event
from datetime import datetime
import logging

# Add the parent directory to sys.path to import from src
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from src.models import (
    NotebookPlanModel,
    Section,
    SubSection,
    NotebookCell,
    NotebookSectionContent,
)
from src.planner import PlannerLLM
from src.writer import WriterAgent
from src.format.plan_format import format_notebook_plan
from src.tools.debug import DebugLevel
from src.utils import configure_logging
from ui.styles import NOTION_STYLE

# Configure logging
configure_logging(debug_level=DebugLevel.DEBUG)

# Get a logger for this module
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="OpenAI Demo Notebook Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add Notion-like styling
st.markdown(NOTION_STYLE, unsafe_allow_html=True)


def display_and_edit_plan(notebook_plan: NotebookPlanModel):
    """Display and allow editing of the notebook plan."""
    logger.info(f"Displaying plan for editing: {notebook_plan.title}")

    # Create a clean, well-organized layout
    st.markdown(
        """
        <div style="padding: 0; margin-bottom: 2rem;">
            <h2 style="color: #2F3437; font-size: 1.8rem; letter-spacing: -0.02em; margin-top: 0.5rem; margin-bottom: 1rem;">
                📝 Notebook Plan
            </h2>
            <p style="color: rgba(55, 53, 47, 0.7); font-size: 1rem; margin-top: 0; margin-bottom: 1.5rem;">
                Define the structure and content of your notebook
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Create columns for the main plan metadata
    col1, col2 = st.columns([2, 1])

    with col1:
        title = st.text_input(
            "Title",
            notebook_plan.title,
            placeholder="Enter a concise, descriptive title",
            help="A clear title helps users understand the notebook's purpose at a glance",
        )

    with col2:
        # Define options list first
        audience_options = [
            "Beginners",
            "Intermediate",
            "Advanced",
            "Experts",
            "Machine Learning Engineers",
            "Data Scientists",
            "Software Engineers",
            "Researchers",
            "Students",
        ]
        # Get index of current audience in the options list, or default to 0
        current_audience_index = (
            audience_options.index(notebook_plan.target_audience)
            if notebook_plan.target_audience in audience_options
            else 0
        )

        target_audience = st.selectbox(
            "Target Audience",
            options=audience_options,
            index=current_audience_index,
            help="Who is this notebook designed for?",
        )

    # Description and purpose
    with st.expander("Description & Purpose", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            description = st.text_area(
                "Description",
                notebook_plan.description,
                height=150,
                placeholder="Provide a comprehensive description of the notebook...",
                help="What does this notebook demonstrate or teach?",
            )

        with col2:
            purpose = st.text_area(
                "Purpose",
                notebook_plan.purpose,
                height=150,
                placeholder="Explain the purpose and goals of this notebook...",
                help="Why would someone use this notebook? What will they learn or accomplish?",
            )

    # Store the edited metadata
    edited_plan = {
        "title": title,
        "description": description,
        "purpose": purpose,
        "target_audience": target_audience,
        "sections": [],
    }

    # Section management
    st.markdown(
        """
        <div style="margin-top: 2rem; margin-bottom: 1rem;">
            <h3 style="color: #2F3437; font-size: 1.4rem; letter-spacing: -0.01em; margin-bottom: 0.5rem;">
                Notebook Sections
            </h3>
            <p style="color: rgba(55, 53, 47, 0.7); font-size: 0.9rem; margin-top: 0; margin-bottom: 1rem;">
                Build your notebook structure by adding, editing, and organizing sections below
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Add section button - make it more visually prominent
    add_col1, add_col2 = st.columns([6, 1])
    with add_col2:
        if st.button("➕ Add Section", type="primary"):
            logger.info("Adding new section")
            new_section = Section(
                title="New Section", description="Enter section description here"
            )
            notebook_plan.sections.append(new_section)
            st.rerun()

    # If there are no sections, show a helpful message
    if not notebook_plan.sections:
        st.info(
            "Your notebook doesn't have any sections yet. Click 'Add Section' to get started."
        )

    # Create a container for the sections
    sections_container = st.container()

    # Display each section with edit capabilities
    with sections_container:
        for idx, section in enumerate(notebook_plan.sections):
            logger.debug(f"Editing section {idx + 1}: {section.title}")

            # Create a card-like container for each section
            st.markdown(
                f"""
                <div style="background: white; border: 1px solid rgba(55, 53, 47, 0.1); border-radius: 4px; 
                      margin-bottom: 16px; overflow: hidden; box-shadow: rgba(15, 15, 15, 0.03) 0px 1px 3px;">
                    <div style="background: #F7F8F9; padding: 12px 16px; border-bottom: 1px solid rgba(55, 53, 47, 0.08);">
                        <h4 style="margin: 0; color: #2F3437; font-size: 1rem;">Section {idx + 1}</h4>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

            with st.expander(f"{section.title}", expanded=True):
                # Section title with improved styling
                section_title = st.text_input(
                    "Section Title",
                    section.title,
                    key=f"section_title_{idx}",
                    placeholder="Enter a clear, concise section title",
                    help="Good section titles help readers navigate your notebook",
                )

                # Section description with improved styling
                section_description = st.text_area(
                    "Section Description",
                    section.description,
                    height=120,
                    key=f"section_description_{idx}",
                    placeholder="Describe what this section covers and its learning objectives...",
                    help="A detailed description helps you plan the content for this section",
                )

                # Subsections area with improved organization
                st.markdown(
                    """
                    <div style="margin: 24px 0 16px 0;">
                        <h5 style="color: #2F3437; font-size: 1rem; margin-bottom: 0.8rem; padding-bottom: 8px; border-bottom: 1px solid rgba(55, 53, 47, 0.1);">
                            Subsections
                        </h5>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

                subsections = []
                if section.subsections:
                    logger.debug(
                        f"Editing {len(section.subsections)} subsections for section {idx + 1}"
                    )

                    # Create a clean table-like layout for subsections
                    for sub_idx, subsection in enumerate(section.subsections):
                        st.markdown(
                            f"""
                            <div style="padding: 12px; background: #F7F8F9; border-radius: 4px; margin-bottom: 14px; border: 1px solid rgba(55, 53, 47, 0.05);">
                                <div style="font-size: 0.95rem; font-weight: 500; color: #2F3437; margin-bottom: 6px;">
                                    Subsection {sub_idx + 1}
                                </div>
                            </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        # Create two columns for title and description
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            sub_title = st.text_input(
                                "Title",
                                subsection.title,
                                key=f"subsection_title_{idx}_{sub_idx}",
                                placeholder="Subsection title",
                            )

                        with col2:
                            sub_description = st.text_area(
                                "Description",
                                subsection.description,
                                key=f"subsection_desc_{idx}_{sub_idx}",
                                height=80,
                                placeholder="Describe the subsection content...",
                            )

                        # Improved delete button with confirmation
                        delete_col1, delete_col2 = st.columns([3, 1])
                        with delete_col2:
                            if st.button(
                                "🗑️ Delete",
                                key=f"delete_subsection_{idx}_{sub_idx}",
                                type="secondary",
                                help="Remove this subsection",
                            ):
                                logger.info(
                                    f"Deleting subsection {sub_idx + 1} from section {idx + 1}"
                                )
                                section.subsections.pop(sub_idx)
                                st.rerun()

                        st.markdown(
                            "<hr style='margin: 15px 0; opacity: 0.2;'>",
                            unsafe_allow_html=True,
                        )
                        subsections.append(
                            SubSection(title=sub_title, description=sub_description)
                        )

                # More elegant add subsection button
                st.markdown(
                    "<div style='margin: 16px 0;'></div>", unsafe_allow_html=True
                )
                if st.button(
                    "➕ Add Subsection",
                    key=f"add_subsection_{idx}",
                    help="Add a new subsection to this section",
                ):
                    logger.info(f"Adding new subsection to section {idx + 1}")
                    if not section.subsections:
                        section.subsections = []
                    section.subsections.append(
                        SubSection(
                            title="New Subsection",
                            description="Enter subsection description",
                        )
                    )
                    st.rerun()

                st.markdown(
                    "<div style='margin: 24px 0 8px 0;'></div>", unsafe_allow_html=True
                )

                # Section controls in a more visually organized footer
                st.markdown(
                    """
                    <div style="background: #F7F8F9; padding: 14px; border-radius: 4px; border: 1px solid rgba(55, 53, 47, 0.05);">
                        <div style="font-size: 0.85rem; color: rgba(55, 53, 47, 0.6); margin-bottom: 10px;">
                            Section Controls
                        </div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

                # Improved section control buttons layout
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

                with col1:
                    if idx > 0:
                        if st.button("⬆️ Move Up", key=f"up_{idx}"):
                            logger.info(f"Moving section {idx + 1} up")
                            (
                                notebook_plan.sections[idx],
                                notebook_plan.sections[idx - 1],
                            ) = (
                                notebook_plan.sections[idx - 1],
                                notebook_plan.sections[idx],
                            )
                            st.rerun()
                    else:
                        st.markdown("&nbsp;")  # Placeholder for empty button

                with col2:
                    if idx < len(notebook_plan.sections) - 1:
                        if st.button("⬇️ Move Down", key=f"down_{idx}"):
                            logger.info(f"Moving section {idx + 1} down")
                            (
                                notebook_plan.sections[idx],
                                notebook_plan.sections[idx + 1],
                            ) = (
                                notebook_plan.sections[idx + 1],
                                notebook_plan.sections[idx],
                            )
                            st.rerun()
                    else:
                        st.markdown("&nbsp;")  # Placeholder for empty button

                with col4:
                    # Dangerous actions should be on the right and stand out less visually
                    if st.button(
                        "🗑️ Delete Section", key=f"delete_{idx}", type="secondary"
                    ):
                        logger.info(f"Deleting section {idx + 1}")
                        notebook_plan.sections.pop(idx)
                        st.rerun()

                # Update the section in the plan
                edited_plan["sections"].append(
                    Section(
                        title=section_title,
                        description=section_description,
                        subsections=subsections if subsections else None,
                    )
                )

    logger.info("Plan editing completed")
    return NotebookPlanModel(**edited_plan)


def create_notebook_from_cells(
    cells: Sequence[Union[NotebookCell, NotebookSectionContent]], title: str
) -> nbformat.NotebookNode:
    """Convert cells to Jupyter notebook format."""
    logger.debug(f"Creating notebook with title: {title}")
    nb = new_notebook()

    # Add title as markdown cell
    nb.cells.append(new_markdown_cell(f"# {title}"))

    # Add cells
    for cell in cells:
        if isinstance(cell, NotebookCell):
            if cell.cell_type == "markdown":
                nb.cells.append(new_markdown_cell(cell.content))
            else:
                nb.cells.append(new_code_cell(cell.content))
        elif isinstance(cell, NotebookSectionContent):
            # Handle section content
            nb.cells.append(new_markdown_cell(f"## {cell.section_title}"))
            for subcell in cell.cells:
                if subcell.cell_type == "markdown":
                    nb.cells.append(new_markdown_cell(subcell.content))
                else:
                    nb.cells.append(new_code_cell(subcell.content))

    logger.debug(f"Created notebook with {len(nb.cells)} cells")
    return nb


def display_notebook_content(
    notebook_content: Sequence[Union[NotebookCell, NotebookSectionContent]], title: str
):
    """Display notebook content in different views."""
    logger.info(f"Displaying notebook content: {title}")

    # Create tabs for different views
    tab_md, tab_code, tab_preview = st.tabs(["Markdown", "Code", "Preview"])

    with tab_md:
        logger.debug("Rendering markdown view")
        # Display pure markdown view
        for cell in notebook_content:
            if isinstance(cell, NotebookCell):
                if cell.cell_type == "markdown":
                    st.markdown(cell.content)
                else:
                    st.code(cell.content, language="python")
            elif isinstance(cell, NotebookSectionContent):
                st.markdown(f"## {cell.section_title}")
                for subcell in cell.cells:
                    if subcell.cell_type == "markdown":
                        st.markdown(subcell.content)
                    else:
                        st.code(subcell.content, language="python")

    with tab_code:
        logger.debug("Rendering code view")
        # Display code-focused view with syntax highlighting
        for cell in notebook_content:
            if isinstance(cell, NotebookCell):
                if cell.cell_type == "code":
                    highlighted_code = highlight(
                        cell.content, PythonLexer(), HtmlFormatter()
                    )
                    st.markdown(
                        f'<div class="highlight">{highlighted_code}</div>',
                        unsafe_allow_html=True,
                    )
            elif isinstance(cell, NotebookSectionContent):
                for subcell in cell.cells:
                    if subcell.cell_type == "code":
                        highlighted_code = highlight(
                            subcell.content, PythonLexer(), HtmlFormatter()
                        )
                        st.markdown(
                            f'<div class="highlight">{highlighted_code}</div>',
                            unsafe_allow_html=True,
                        )

    with tab_preview:
        logger.debug("Rendering notebook preview")
        # Create and display notebook preview
        notebook = create_notebook_from_cells(notebook_content, title)
        # Convert notebook to HTML
        from nbconvert import HTMLExporter

        html_exporter = HTMLExporter()
        html_data, _ = html_exporter.from_notebook_node(notebook)
        components.html(html_data, height=600, scrolling=True)

    logger.debug("Notebook content display completed")


def ui_clarification_callback(questions: List[str]) -> Dict[str, str]:
    """
    Callback function for handling clarifications in the UI.
    Stores questions in session state and signals that clarification is needed.

    Note: This is designed for the async nature of Streamlit - returns empty dict
    initially and the actual answers will be processed in the next Streamlit run.
    """
    # Store questions in session state
    st.session_state.clarification_questions = questions
    st.session_state.needs_clarification = True

    # Return an empty dict - actual answers will be processed in the next run
    return {}


def handle_clarifications():
    """Display clarification questions and collect answers."""
    logger.info("Handling clarification questions")
    questions = st.session_state.clarification_questions

    st.info("The AI needs some clarifications before proceeding:")

    # Create a form for the questions
    with st.form("clarification_form"):
        st.markdown("### Please answer these questions:")

        # Dictionary to store answers
        answers = {}

        # Display each question with a text input
        for i, question in enumerate(questions):
            answer = st.text_area(
                f"Question {i+1}: {question}",
                value="",
                key=f"clarification_{i}",
            )
            answers[question] = answer

        # Submit button with centered styling
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        submit_pressed = st.form_submit_button("Submit Answers")
        st.markdown("</div>", unsafe_allow_html=True)

        if submit_pressed:
            logger.info("Clarification answers submitted")
            if all(answers.values()):  # Check if all questions are answered
                # Store answers in the all_clarifications dictionary
                st.session_state.all_clarifications = answers

                # Get the existing planner
                planner = st.session_state.planner_state
                if planner:
                    try:
                        with st.spinner(
                            "Processing clarifications and updating plan..."
                        ):
                            # Call the planner with the clarification answers
                            result = planner.plan_notebook_with_ui(
                                requirements=st.session_state.user_requirements,
                                previous_messages=st.session_state.previous_messages,
                                clarification_answers=answers,
                            )

                            # Check if we need more clarifications or have a plan
                            if isinstance(result, list):
                                # More questions needed
                                st.session_state.clarification_questions = result
                                st.session_state.previous_messages = (
                                    planner.get_current_messages()
                                )
                                st.rerun()
                            else:
                                # We have a complete plan
                                st.session_state.notebook_plan = result
                                st.session_state.step = 2
                                st.session_state.needs_clarification = False
                                st.session_state.planner_state = None
                                st.session_state.previous_messages = None
                                st.session_state.all_clarifications = {}
                                st.rerun()
                    except Exception as e:
                        logger.error(f"Error processing clarifications: {str(e)}")
                        st.error(f"Error processing clarifications: {str(e)}")
                        return
            else:
                logger.warning("Not all clarification questions were answered")
                st.error("Please answer all questions before proceeding.")


def save_session_state():
    """Save current session state to a file."""
    logger.info("Saving session state")
    session_data = {
        "user_requirements": st.session_state.user_requirements,
        "notebook_plan": (
            st.session_state.notebook_plan.model_dump()
            if st.session_state.notebook_plan
            else None
        ),
        "step": st.session_state.step,
    }

    # Create output/sessions directory if it doesn't exist
    output_session_dir = "output/sessions"
    os.makedirs(output_session_dir, exist_ok=True)

    # Save with timestamp inside output/sessions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{timestamp}.json"
    file_path = os.path.join(output_session_dir, filename)

    with open(file_path, "w") as f:
        json.dump(session_data, f)

    logger.info(f"Session state saved to {file_path}")
    return file_path


def load_session_state(file_path: str):
    """Load session state from a file."""
    logger.info(f"Loading session state from {file_path}")
    try:
        with open(file_path, "r") as f:
            session_data = json.load(f)

        st.session_state.user_requirements = session_data["user_requirements"]
        if session_data["notebook_plan"]:
            st.session_state.notebook_plan = NotebookPlanModel(
                **session_data["notebook_plan"]
            )
        st.session_state.step = session_data["step"]
        logger.info("Session state loaded successfully")
        return True
    except FileNotFoundError:
        logger.error(f"Session file not found: {file_path}")
        st.error(f"Error: Session file not found at {file_path}")
        return False
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from session file: {file_path}")
        st.error("Error: Could not read the session file. It might be corrupted.")
        return False
    except Exception as e:
        logger.error(f"Error loading session state: {str(e)}")
        st.error(f"An unexpected error occurred while loading the session: {str(e)}")
        return False


def display_file_management():
    """Display file management section with download options and session management."""
    logger.info("Displaying file management section")

    st.markdown(
        """
        <div style="margin: 40px 0 20px 0;">
            <h3 style="color: #2F3437; font-size: 1.4rem; letter-spacing: -0.01em; margin-bottom: 0.5rem;">
                📁 File Management
            </h3>
            <p style="color: rgba(55, 53, 47, 0.7); font-size: 0.9rem; margin-top: 0; margin-bottom: 1.5rem;">
                Save and download files, or load a previous session
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # File management in a Notion-like card
    with st.container():
        # Add a Notion-like card styling
        st.markdown('<div class="file-management">', unsafe_allow_html=True)

        # Download files section
        st.markdown(
            """
            <h4 style="color: #2F3437; font-size: 1.2rem; margin-top: 0;">
                Download Files
            </h4>
            <p style="color: rgba(55, 53, 47, 0.7); font-size: 0.9rem; margin-bottom: 1.2rem;">
                Download your notebook in various formats
            </p>
        """,
            unsafe_allow_html=True,
        )

        # Create a clean grid for download buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.session_state.notebook_plan:
                # Define output directory and ensure it exists
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)

                # Save plan to file inside output directory
                plan_file_name = (
                    f"{st.session_state.notebook_plan.title.replace(' ', '_')}_plan.md"
                )
                plan_file_path = os.path.join(output_dir, plan_file_name)

                try:
                    with open(plan_file_path, "w") as f:
                        f.write(format_notebook_plan(st.session_state.notebook_plan))
                    logger.debug(
                        f"Plan temporarily saved to {plan_file_path} for download"
                    )

                    with open(plan_file_path, "r") as f:
                        st.download_button(
                            "📄 Download Plan",
                            f.read(),
                            file_name=plan_file_name,  # Use only filename for download button
                            mime="text/markdown",
                            help="Download the notebook plan in Markdown format",
                            use_container_width=True,
                        )
                    # Clean up the temporary file after creating the button
                    # os.remove(plan_file_path)
                    # logger.debug(f"Removed temporary plan file: {plan_file_path}")
                    # Note: Keeping the file might be useful for persistence/debugging
                    # If removal is desired, uncomment the lines above.
                except Exception as e:
                    logger.error(f"Error preparing plan for download: {e}")
                    st.error("Could not prepare plan for download.")

        with col2:
            if st.session_state.notebook_content and st.session_state.notebook_plan:
                notebook = create_notebook_from_cells(
                    st.session_state.notebook_content,
                    st.session_state.notebook_plan.title,
                )
                notebook_json = nbformat.writes(notebook)
                st.download_button(
                    "📓 Download Notebook",
                    notebook_json,
                    file_name=f"{st.session_state.notebook_plan.title.replace(' ', '_')}.ipynb",
                    mime="application/x-ipynb+json",
                    help="Download the generated Jupyter notebook",
                    use_container_width=True,
                )
                logger.debug("Added notebook download button")

        with col3:
            if st.session_state.notebook_content:
                markdown_content = []
                for cell in st.session_state.notebook_content:
                    if isinstance(cell, NotebookCell):
                        markdown_content.append(cell.content)
                    elif isinstance(cell, NotebookSectionContent):
                        markdown_content.append(f"## {cell.section_title}")
                        for subcell in cell.cells:
                            markdown_content.append(subcell.content)

                st.download_button(
                    "📝 Download Markdown",
                    "\n\n".join(markdown_content),
                    file_name=f"{st.session_state.notebook_plan.title.replace(' ', '_')}.md",
                    mime="text/markdown",
                    help="Download the notebook content in Markdown format",
                    use_container_width=True,
                )
                logger.debug("Added markdown download button")

        # Session management
        st.markdown(
            """
            <div style="margin: 30px 0 15px 0;">
                <h4 style="color: #2F3437; font-size: 1.2rem; margin-bottom: 0.5rem;">
                    Session Management
                </h4>
                <p style="color: rgba(55, 53, 47, 0.7); font-size: 0.9rem; margin-bottom: 1.2rem;">
                    Save your progress or load a previous session
                </p>
            </div>
        """,
            unsafe_allow_html=True,
        )

        # Use columns for better organization
        col1, col2 = st.columns(2)

        with col1:
            # Save session button with improved styling
            if st.button("💾 Save Session", type="primary", use_container_width=True):
                saved_file = save_session_state()
                logger.info(f"Session saved to {saved_file}")
                st.success(f"Session saved to {saved_file}")

        with col2:
            # File uploader with improved styling
            uploaded_file = st.file_uploader(
                "Load Session",
                type="json",
                help="Upload a previously saved session file (from output/sessions/)",
                label_visibility="collapsed",
            )
            if uploaded_file:
                # Define output directory for temporary storage
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                temp_file_path = os.path.join(output_dir, "temp_session.json")

                # Save uploaded file temporarily
                try:
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    logger.info(
                        f"Uploaded session saved temporarily to {temp_file_path}"
                    )

                    if load_session_state(temp_file_path):
                        st.success("Session loaded successfully!")
                        st.rerun()
                    # No need for else here, load_session_state handles errors
                except Exception as e:
                    logger.error(f"Error handling uploaded session file: {e}")
                    st.error(f"Failed to process uploaded session file: {e}")
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                            logger.info(
                                f"Removed temporary session file: {temp_file_path}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Error removing temporary session file {temp_file_path}: {e}"
                            )

        st.markdown("</div>", unsafe_allow_html=True)
        logger.debug("File management section display completed")


def main():
    # Clean, minimal header with Notion-like styling
    st.markdown(
        """
        <div style="padding: 1.5rem 0; margin-bottom: 2rem;">
            <h1 style="margin-bottom: 0.5rem; font-size: 2.5rem; font-weight: 700; letter-spacing: -0.05em;">
                📚 Notebook Generator
            </h1>
            <p style="color: rgba(55, 53, 47, 0.7); font-size: 1.1rem; margin-top: 0; font-weight: 400;">
                Create beautiful Python notebooks showcasing OpenAI API capabilities with ease.
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    logger.info("Starting OpenAI Demo Notebook Generator")

    # Initialize session state variables
    if "step" not in st.session_state:
        st.session_state.step = 1
        logger.debug("Initialized step to 1")
    if "user_requirements" not in st.session_state:
        st.session_state.user_requirements = {}
        logger.debug("Initialized user_requirements")
    if "notebook_plan" not in st.session_state:
        st.session_state.notebook_plan = None
        logger.debug("Initialized notebook_plan")
    if "notebook_content" not in st.session_state:
        st.session_state.notebook_content = None
        logger.debug("Initialized notebook_content")
    if "needs_clarification" not in st.session_state:
        st.session_state.needs_clarification = False
        logger.debug("Initialized needs_clarification")
    if "clarification_questions" not in st.session_state:
        st.session_state.clarification_questions = []
        logger.debug("Initialized clarification_questions")
    if "all_clarifications" not in st.session_state:
        st.session_state.all_clarifications = {}
        logger.debug("Initialized all_clarifications")
    if "planner_state" not in st.session_state:
        st.session_state.planner_state = None
        logger.debug("Initialized planner_state")

    logger.info(f"Current step: {st.session_state.step}")

    # Create a clean progress indicator
    steps = ["Requirements", "Plan", "Notebook"]
    current_step = st.session_state.step - 1  # 0-based index for steps list

    cols = st.columns(len(steps))
    for i, step in enumerate(steps):
        with cols[i]:
            if i < current_step:
                # Completed step
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 10px; color: #2F3437;">
                        <div style="background: #2F3437; color: white; width: 28px; height: 28px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 8px; font-weight: 600;">✓</div>
                        <div style="font-size: 0.85rem; font-weight: 500; opacity: 0.7;">{step}</div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
            elif i == current_step:
                # Current step
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 10px; color: #2F3437;">
                        <div style="background: #2F3437; color: white; width: 28px; height: 28px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 8px; font-weight: 600;">{i+1}</div>
                        <div style="font-size: 0.85rem; font-weight: 600;">{step}</div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                # Upcoming step
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 10px; color: rgba(55, 53, 47, 0.4);">
                        <div style="background: #F7F8F9; color: rgba(55, 53, 47, 0.4); width: 28px; height: 28px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 8px; border: 1px solid rgba(55, 53, 47, 0.1); font-weight: 600;">{i+1}</div>
                        <div style="font-size: 0.85rem; font-weight: 500;">{step}</div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

    # Add a subtle divider
    st.markdown(
        "<hr style='margin-top: 0; margin-bottom: 2rem; opacity: 0.6;'>",
        unsafe_allow_html=True,
    )

    # Step 1: Input Form
    if st.session_state.step == 1:
        logger.info("Displaying input form (Step 1)")

        # Use a Notion-like card for the form
        st.markdown(
            """
            <div class="notion-card" style="padding: 28px; margin-bottom: 24px;">
                <h2 style="margin-top: 0; color: #2F3437; font-size: 1.5rem; margin-bottom: 1.5rem;">Notebook Requirements</h2>
            </div>
        """,
            unsafe_allow_html=True,
        )

        with st.form("notebook_requirements"):
            # Notebook Description
            description = st.text_area(
                "Notebook Description",
                help="Describe what you want to demonstrate in this notebook",
                placeholder="Example: Create a notebook demonstrating how to use OpenAI's GPT-4 for text summarization",
                key="description",
                height=120,
            )

            # Create two columns for audience and additional requirements
            col1, col2 = st.columns([1, 1])

            with col1:
                # Target Audience
                audience = st.selectbox(
                    "Target Audience",
                    options=[
                        "Beginners",
                        "Intermediate",
                        "Advanced",
                        "Experts",
                        "Machine Learning Engineers",
                        "Data Scientists",
                        "Software Engineers",
                        "Researchers",
                        "Students",
                    ],
                    help="Select the technical level of your target audience",
                    key="audience",
                )

            with col2:
                # Additional Requirements
                additional_reqs = st.text_area(
                    "Additional Requirements (Optional)",
                    help="Any specific requirements or features you want to include",
                    placeholder="Example: Include error handling, Add visualization of results",
                    key="additional_reqs",
                    height=124,
                )

            # Submit button with centered styling
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Generate Plan")
            st.markdown("</div>", unsafe_allow_html=True)

            if submitted:
                # Rest of the submitted logic remains unchanged
                logger.info("Form submitted - validating inputs")
                # Validate inputs
                if not description:
                    logger.warning("No description provided")
                    st.error("Please provide a notebook description.")
                    return

                # Store in session state
                st.session_state.user_requirements = {
                    "notebook_description": description,
                    "target_audience": audience,
                    "additional_requirements": [
                        req.strip()
                        for req in additional_reqs.split("\n")
                        if req.strip()
                    ],
                }
                logger.info("User requirements stored in session state")

                # Initialize planner if not already initialized
                if not st.session_state.planner_state:
                    logger.info("Initializing new planner")
                    planner = PlannerLLM()
                    st.session_state.planner_state = planner
                else:
                    logger.debug("Using existing planner from state")
                    planner = st.session_state.planner_state

                try:
                    with st.spinner("Analyzing requirements and generating plan..."):
                        logger.info("Starting plan generation")
                        # Check if we need to handle clarifications
                        if st.session_state.needs_clarification:
                            logger.info("Waiting for clarification answers")
                            # We have questions but no answers yet - wait for user input
                            pass
                        else:
                            # Store messages in session state if not already present
                            if "previous_messages" not in st.session_state:
                                st.session_state.previous_messages = None

                            # Get clarification answers if available
                            clarification_answers = (
                                st.session_state.all_clarifications
                                if st.session_state.all_clarifications
                                else None
                            )

                            # Call the UI-friendly planning method
                            logger.info("Calling planner to generate notebook plan")
                            result = planner.plan_notebook_with_ui(
                                requirements=st.session_state.user_requirements,
                                previous_messages=st.session_state.previous_messages,
                                clarification_answers=clarification_answers,
                            )

                            # Check if the result is a list of questions or a plan
                            if isinstance(result, list):
                                logger.info(
                                    "Received clarification questions from planner"
                                )
                                # Store questions and current message history
                                st.session_state.clarification_questions = result
                                st.session_state.previous_messages = (
                                    planner.get_current_messages()
                                )
                                st.session_state.needs_clarification = True
                                st.rerun()
                            else:
                                logger.info("Received complete notebook plan")
                                # We have a complete plan
                                st.session_state.notebook_plan = result
                                st.session_state.step = 2

                                # Clear planner state and message history now that we're done
                                st.session_state.planner_state = None
                                st.session_state.previous_messages = None
                                st.session_state.needs_clarification = False
                                st.session_state.all_clarifications = {}
                                logger.debug("Cleared temporary planning state")
                                st.rerun()

                except Exception as e:
                    logger.error(f"Error during planning: {str(e)}")
                    st.error(f"Error during planning: {str(e)}")
                    # Clear planner state on error
                    st.session_state.planner_state = None
                    return

    # Step 1.5: Clarifications (if needed)
    if st.session_state.needs_clarification:
        logger.info("Handling clarification questions (Step 1.5)")
        handle_clarifications()

    # Step 2: Plan Display and Editing
    elif st.session_state.step == 2 and st.session_state.notebook_plan:
        logger.info("Displaying plan for editing (Step 2)")
        # Back button
        if st.button("← Back to Requirements"):
            logger.info("User clicked back to requirements")
            st.session_state.step = 1
            st.rerun()

        # Display and edit the plan
        edited_plan = display_and_edit_plan(st.session_state.notebook_plan)
        st.session_state.notebook_plan = edited_plan
        logger.debug("Updated notebook plan with edits")

        # Continue button
        if st.button("Generate Notebook"):
            logger.info("User clicked generate notebook")
            st.session_state.step = 3
            st.rerun()

    # Step 3: Content Generation and Display
    elif st.session_state.step == 3:
        logger.info("Starting content generation (Step 3)")
        # Back button
        if st.button("← Back to Plan"):
            logger.info("User clicked back to plan")
            st.session_state.step = 2
            st.rerun()

        # Generate content if not already generated
        if not st.session_state.notebook_content:
            with st.spinner("Generating notebook content..."):
                logger.info("Initializing content generation")
                # Initialize progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Initialize writer agent
                    logger.info("Initializing writer agent")
                    writer = WriterAgent(
                        model="gpt-4o",
                        max_retries=3,
                        search_enabled=True,
                        final_critique_enabled=True,
                    )

                    # Update progress
                    progress_bar.progress(20)
                    status_text.text(
                        "Analyzing plan and preparing content structure..."
                    )
                    logger.info("Starting content generation")

                    # Generate content
                    result = writer.generate_content(
                        notebook_plan=cast(
                            NotebookPlanModel, st.session_state.notebook_plan
                        ),
                        additional_requirements=st.session_state.user_requirements.get(
                            "additional_requirements", []
                        ),
                    )
                    print("result", result)

                    # Update progress
                    progress_bar.progress(60)
                    status_text.text("Reviewing and refining generated content...")
                    logger.info("Content generation completed, processing results")

                    # Handle the result (which might be a tuple with critique)
                    if isinstance(result, tuple) and len(result) == 3:
                        logger.info("Processing content with critique")
                        original_content, critique, revised_content = result
                        st.session_state.notebook_content = revised_content

                        # Store critique for display
                        st.session_state.content_critique = critique
                    else:
                        logger.info("Processing content without critique")
                        st.session_state.notebook_content = result

                    # Update progress
                    progress_bar.progress(100)
                    status_text.text("Content generation complete!")
                    logger.info(
                        "Content generation and processing completed successfully"
                    )

                except Exception as e:
                    logger.error(f"Error generating notebook content: {str(e)}")
                    st.error(f"Error generating notebook content: {str(e)}")
                    return

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

        # Display the generated content
        if st.session_state.notebook_content and st.session_state.notebook_plan:
            logger.info("Displaying generated notebook content")
            st.subheader("Generated Notebook")

            # Display critique if available
            if hasattr(st.session_state, "content_critique"):
                logger.debug("Displaying content critique")
                with st.expander("Content Critique", expanded=False):
                    st.markdown(st.session_state.content_critique)

            # Display the notebook content
            display_notebook_content(
                st.session_state.notebook_content, st.session_state.notebook_plan.title
            )

            # Add download buttons
            st.subheader("Download Options")
            col1, col2 = st.columns(2)

            with col1:
                # Create .ipynb file for download
                notebook = create_notebook_from_cells(
                    st.session_state.notebook_content,
                    st.session_state.notebook_plan.title,
                )
                notebook_json = nbformat.writes(notebook)
                st.download_button(
                    "Download Jupyter Notebook",
                    notebook_json,
                    file_name=f"{st.session_state.notebook_plan.title.replace(' ', '_')}.ipynb",
                    mime="application/x-ipynb+json",
                )
                logger.debug("Added Jupyter notebook download button")

            with col2:
                # Create markdown version for download
                markdown_content = []
                for cell in st.session_state.notebook_content:
                    if isinstance(cell, NotebookCell):
                        markdown_content.append(cell.content)
                    elif isinstance(cell, NotebookSectionContent):
                        markdown_content.append(f"## {cell.section_title}")
                        for subcell in cell.cells:
                            markdown_content.append(subcell.content)

                st.download_button(
                    "Download Markdown",
                    "\n\n".join(markdown_content),
                    file_name=f"{st.session_state.notebook_plan.title.replace(' ', '_')}.md",
                    mime="text/markdown",
                )
                logger.debug("Added Markdown download button")

    # Add file management section at the bottom of each step after step 1
    if st.session_state.step > 1:
        logger.debug("Displaying file management section")
        display_file_management()

    st.markdown("</div>", unsafe_allow_html=True)
    logger.debug("Page rendering completed")


if __name__ == "__main__":
    main()
