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
st.markdown(
    """
<style>
    /* Global Notion-like styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main container styles */
    .main {
        font-family: 'Inter', sans-serif;
        color: rgb(55, 53, 47);
        line-height: 1.5;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: rgb(55, 53, 47);
        margin-bottom: 1rem;
    }
    
    /* Text areas and inputs */
    .stTextArea textarea, .stTextInput input {
        font-family: 'Inter', sans-serif;
        border-radius: 3px;
        border: 1px solid rgba(55, 53, 47, 0.16);
        padding: 12px;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: rgb(45, 170, 219);
        box-shadow: rgba(45, 170, 219, 0.3) 0px 0px 0px 2px;
    }
    
    /* Buttons */
    .stButton button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        background: rgb(45, 170, 219);
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 3px;
        transition: background 0.2s;
    }
    
    .stButton button:hover {
        background: rgb(35, 131, 226);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: rgb(55, 53, 47);
        background: rgba(55, 53, 47, 0.03);
        border-radius: 3px;
    }
    
    /* Cards */
    .notion-card {
        background: white;
        border: 1px solid rgba(55, 53, 47, 0.16);
        border-radius: 3px;
        padding: 16px;
        margin: 8px 0;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: rgb(45, 170, 219);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
        background-color: rgba(55, 53, 47, 0.05);
        padding: 0px;
        border-radius: 3px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: transparent;
        border: none;
        color: rgb(55, 53, 47);
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(55, 53, 47, 0.05);
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(45, 170, 219, 0.1);
        color: rgb(45, 170, 219);
    }
    
    /* Code blocks */
    .highlight {
        background: rgb(247, 246, 243);
        border-radius: 3px;
        padding: 16px;
        margin: 8px 0;
    }
    
    /* File management section */
    .file-management {
        background: white;
        border: 1px solid rgba(55, 53, 47, 0.16);
        border-radius: 3px;
        padding: 20px;
        margin: 20px 0;
    }
    
    .download-button {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 8px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


def display_and_edit_plan(notebook_plan: NotebookPlanModel):
    """Display and allow editing of the notebook plan."""
    logger.info(f"Displaying plan for editing: {notebook_plan.title}")
    st.subheader("Notebook Plan")

    # Display plan sections in an editable format
    edited_plan = {
        "title": st.text_input("Title", notebook_plan.title),
        "description": st.text_area("Description", notebook_plan.description),
        "purpose": st.text_area("Purpose", notebook_plan.purpose),
        "target_audience": notebook_plan.target_audience,
        "sections": [],
    }

    st.subheader("Sections")
    logger.debug(f"Displaying {len(notebook_plan.sections)} sections for editing")

    # Create a container for the sections
    sections_container = st.container()

    # Add section button
    if st.button("Add New Section"):
        logger.info("Adding new section")
        new_section = Section(
            title="New Section", description="Enter section description here"
        )
        notebook_plan.sections.append(new_section)

    # Display each section with edit capabilities
    with sections_container:
        for idx, section in enumerate(notebook_plan.sections):
            logger.debug(f"Editing section {idx + 1}: {section.title}")
            with st.expander(f"Section {idx + 1}: {section.title}", expanded=True):
                # Section title
                section_title = st.text_input(
                    f"Section Title ###{idx}", section.title, key=f"section_title_{idx}"
                )

                # Section description
                section_description = st.text_area(
                    f"Section Description ###{idx}",
                    section.description,
                    height=150,
                    key=f"section_description_{idx}",
                )

                # Subsections
                st.markdown("---")  # Visual separator
                st.subheader("Subsections")
                subsections = []
                if section.subsections:
                    logger.debug(
                        f"Editing {len(section.subsections)} subsections for section {idx + 1}"
                    )
                    for sub_idx, subsection in enumerate(section.subsections):
                        st.markdown(f"##### Subsection {sub_idx + 1}")

                        # Create two columns for title and description
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            sub_title = st.text_input(
                                "Title",
                                subsection.title,
                                key=f"subsection_title_{idx}_{sub_idx}",
                            )

                        with col2:
                            sub_description = st.text_area(
                                "Description",
                                subsection.description,
                                key=f"subsection_desc_{idx}_{sub_idx}",
                                height=100,
                            )

                        # Add delete button for subsection
                        if st.button(
                            "Delete Subsection",
                            key=f"delete_subsection_{idx}_{sub_idx}",
                        ):
                            logger.info(
                                f"Deleting subsection {sub_idx + 1} from section {idx + 1}"
                            )
                            section.subsections.pop(sub_idx)
                            st.rerun()

                        st.markdown("---")  # Visual separator between subsections
                        subsections.append(
                            SubSection(title=sub_title, description=sub_description)
                        )

                # Add new subsection button
                if st.button(f"Add Subsection", key=f"add_subsection_{idx}"):
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

                st.markdown("---")  # Visual separator

                # Section controls
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Move Up", key=f"up_{idx}") and idx > 0:
                        logger.info(f"Moving section {idx + 1} up")
                        notebook_plan.sections[idx], notebook_plan.sections[idx - 1] = (
                            notebook_plan.sections[idx - 1],
                            notebook_plan.sections[idx],
                        )
                        st.rerun()
                with col2:
                    if (
                        st.button("Move Down", key=f"down_{idx}")
                        and idx < len(notebook_plan.sections) - 1
                    ):
                        logger.info(f"Moving section {idx + 1} down")
                        notebook_plan.sections[idx], notebook_plan.sections[idx + 1] = (
                            notebook_plan.sections[idx + 1],
                            notebook_plan.sections[idx],
                        )
                        st.rerun()
                with col3:
                    if st.button("Delete Section", key=f"delete_{idx}"):
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

        # Submit button
        submit_pressed = st.form_submit_button("Submit Answers")

        if submit_pressed:
            logger.info("Clarification answers submitted")
            if all(answers.values()):  # Check if all questions are answered
                # Store answers in the all_clarifications dictionary
                st.session_state.all_clarifications = answers
                st.session_state.needs_clarification = False
                logger.debug(
                    f"Collected answers: {st.session_state.all_clarifications}"
                )
                st.rerun()
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

    # Create sessions directory if it doesn't exist
    os.makedirs("sessions", exist_ok=True)

    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sessions/session_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(session_data, f)

    logger.info(f"Session state saved to {filename}")
    return filename


def load_session_state(file_path: str):
    """Load session state from a file."""
    logger.info(f"Loading session state from {file_path}")
    with open(file_path, "r") as f:
        session_data = json.load(f)

    st.session_state.user_requirements = session_data["user_requirements"]
    if session_data["notebook_plan"]:
        st.session_state.notebook_plan = NotebookPlanModel(
            **session_data["notebook_plan"]
        )
    st.session_state.step = session_data["step"]
    logger.info("Session state loaded successfully")


def display_file_management():
    """Display file management section with download options and session management."""
    logger.info("Displaying file management section")
    st.markdown("### 📁 File Management")

    with st.container():
        st.markdown('<div class="file-management">', unsafe_allow_html=True)

        # Download files section
        st.subheader("Download Files")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.session_state.notebook_plan:
                # Save plan to file
                plan_file = (
                    f"{st.session_state.notebook_plan.title.replace(' ', '_')}_plan.md"
                )
                with open(plan_file, "w") as f:
                    f.write(format_notebook_plan(st.session_state.notebook_plan))

                with open(plan_file, "r") as f:
                    st.download_button(
                        "📄 Download Plan",
                        f.read(),
                        file_name=plan_file,
                        mime="text/markdown",
                        help="Download the notebook plan in Markdown format",
                    )
                logger.debug("Added plan download button")

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
                )
                logger.debug("Added markdown download button")

        # Session management
        st.subheader("Session Management")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save Session"):
                saved_file = save_session_state()
                logger.info(f"Session saved to {saved_file}")
                st.success(f"Session saved to {saved_file}")

        with col2:
            uploaded_file = st.file_uploader(
                "Load Session",
                type="json",
                help="Upload a previously saved session file",
            )
            if uploaded_file:
                # Save uploaded file temporarily
                with open("temp_session.json", "wb") as f:
                    f.write(uploaded_file.getvalue())
                load_session_state("temp_session.json")
                os.remove("temp_session.json")
                logger.info("Session loaded from uploaded file")
                st.success("Session loaded successfully!")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        logger.debug("File management section display completed")


def main():
    # Header with Notion-like styling
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.title("📚 OpenAI Demo Notebook Generator")
    st.markdown(
        """
    Create beautiful Python notebooks showcasing OpenAI API capabilities with ease.
    Fill in the form below to get started.
    """
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

    # Step 1: Input Form
    if st.session_state.step == 1:
        logger.info("Displaying input form (Step 1)")
        with st.form("notebook_requirements"):
            st.subheader("Notebook Requirements")

            # Notebook Description
            description = st.text_area(
                "Notebook Description",
                help="Describe what you want to demonstrate in this notebook",
                placeholder="Example: Create a notebook demonstrating how to use OpenAI's GPT-4 for text summarization",
                key="description",
            )

            # Target Audience
            audience = st.selectbox(
                "Target Audience",
                options=["Beginners", "Intermediate", "Advanced"],
                help="Select the technical level of your target audience",
                key="audience",
            )

            # Additional Requirements
            additional_reqs = st.text_area(
                "Additional Requirements (Optional)",
                help="Any specific requirements or features you want to include",
                placeholder="Example: Include error handling, Add visualization of results",
                key="additional_reqs",
            )

            # Submit button
            submitted = st.form_submit_button("Generate Plan")

            if submitted:
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
