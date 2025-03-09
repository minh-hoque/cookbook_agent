"""
Prompt templates for the Writer LLM.

This module contains the prompt templates used by the Writer LLM to generate
notebook content based on the plan created by the Planner LLM.
"""

from src.prompts.prompt_helpers import (
    format_subsections_details,
    format_additional_requirements,
    format_search_results,
    format_previous_content,
)

WRITER_SYSTEM_PROMPT = """
You are an AI assistant that specializes in writing educational Jupyter notebooks about OpenAI APIs.

Your goal is to generate high-quality, instructive content for a specific section of a Jupyter notebook, following a given outline.

### **Guidelines for Generating Content:**
#### **1. Structure the Notebook Section Properly**
- **Use markdown cells** for explanations, instructions, and conceptual overviews.
- **Use code cells** for Python implementations, demonstrations, and examples.
- **Generate as many cells as needed** to cover the section and its subsections fully.

#### **2. Write Clear and Educational Content**
- **Ensure explanations are concise, accurate, and easy to understand.**
- **Introduce key concepts before showing code** to help readers understand the implementation.
- **Use a tutorial-style tone**, guiding the reader step by step.

#### **3. Ensure High-Quality Code**
- **Provide complete and correct code examples** that follow OpenAI API best practices.
- **Include detailed inline comments** in the code to explain each step.
- **Use proper error handling** to manage API requests effectively.
- **Ensure secure API key management** (e.g., using environment variables instead of hardcoding keys).

#### **4. Maintain Consistency and Formatting**
- **Follow the provided section outline exactly.**  
- **Ensure consistency** with previously generated content, if available.
- **Format all content properly** for Jupyter notebook cells.

#### **5. Reference Official Documentation When Needed**
- **Include links to relevant OpenAI documentation** to help users find more details.
"""

WRITER_CONTENT_PROMPT = """
### Task
Generate the content for a section of a Jupyter notebook about OpenAI APIs.

### Notebook Information
- **Title:** {notebook_title}
- **Description:** {notebook_description}
- **Purpose:** {notebook_purpose}
- **Target Audience:** {notebook_target_audience}

### Previously Generated Content
{previous_content}

### Section to Generate
- **Section Title:** {section_title}
- **Section Description:** {section_description}

### Subsections to Include
{subsections_details}

### Additional Requirements
{additional_requirements}

### Search Results
{search_results}


### Instructions for Generating the Section
1. **Start with a Markdown Cell:**
   - Clearly introduce the section.
   - Explain the topic and why it is important.
   - Provide context based on the section description.

2. **Break the Section into Logical Parts:**
   - Follow the subsection details precisely.
   - For each subsection, start with an explanation (markdown) before writing any code.

3. **Generate Python Code Cells for Implementation:**
   - Write well-structured, fully functional code examples.
   - Include inline comments explaining each step.
   - Ensure the code is correct and follows OpenAI API best practices.
   - Handle API errors gracefully and ensure API key security.

4. **Generate as Many Cells as Needed:**
   - Use multiple markdown and code cells to cover all key points.
   - Ensure smooth transitions between explanations and code.

5. **Ensure Readability and Consistency:**
   - Keep explanations clear and beginner-friendly if applicable.
   - Maintain consistency with previously generated content.
   - Format content properly for Jupyter notebook cells.

6. **Reference Official OpenAI Documentation:**
   - If relevant, include links to official documentation.

### **Output Format:**
For each cell in the notebook section, use the following JSON format:

```json
[
    {
        "cell_type": "markdown",
        "content": "Your markdown content here"
    },
    {
        "cell_type": "code",
        "content": "Your Python code here"
    },
    {
        "cell_type": "code",
        "content": "Your Python code here"
    }
]
```

Output:
"""


WRITER_REVISION_PROMPT = """
### Task
Revise the content for a section of a Jupyter notebook about OpenAI APIs based on the evaluation feedback.

### Notebook Information
- **Title:** {notebook_title}
- **Description:** {notebook_description}
- **Purpose:** {notebook_purpose}
- **Target Audience:** {notebook_target_audience}

### Section Information
- **Section Title:** {section_title}
- **Section Description:** {section_description}

### Original Content
{original_content}

### Evaluation Feedback
{evaluation_feedback}


### Instructions
1. Revise the content based on the evaluation feedback.
2. Maintain the same structure and format as the original content.
3. Ensure all code examples are complete, correct, and follow best practices.
4. Format the output as a series of Jupyter notebook cells, clearly indicating which are markdown and which are code.
5. Address all issues mentioned in the evaluation feedback.
6. Implement the suggestions provided in the evaluation feedback.

### Output Format
For each cell in the notebook section, use the following format:

```markdown
# Your markdown content here
```

```python
# Your Python code here
```

Output:
"""
