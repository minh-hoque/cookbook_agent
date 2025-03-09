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

Your task is to generate high-quality, instructive content for a specific section of a Jupyter notebook based on the provided outline.

### **Key Guidelines:**
- Write clear, concise, and educational content that follows the section outline exactly.
- Include both markdown explanatory text and Python code cells.
- Ensure code examples are complete, correct, and follow best practices for OpenAI API usage.
- Add detailed comments in code to explain each step.
- Include proper error handling and API key management in code examples.
- Reference official OpenAI documentation where appropriate.
- Maintain a tutorial-like tone that guides the reader through concepts and implementation.
- Focus on creating practical, runnable examples that demonstrate the concepts effectively.
- Ensure all code is properly formatted for Jupyter notebook cells.
"""

WRITER_CONTENT_PROMPT = """
### **Task:**  
Generate the content for a section of a Jupyter notebook about OpenAI APIs.

### **Notebook Information:**
- **Title:** {notebook_title}
- **Description:** {notebook_description}
- **Purpose:** {notebook_purpose}
- **Target Audience:** {notebook_target_audience}

### **Previously Generated Content:**
{previous_content}

### **Section to Generate:**
- **Section Title:** {section_title}
- **Section Description:** {section_description}

### **Subsections to Include:**
{subsections_details}

### **Additional Requirements:**
{additional_requirements}

### **Search Results:**
{search_results}

### **Instructions:**
1. Generate both markdown text and Python code cells for this section.
2. Follow the section description and subsection details precisely.
3. Include clear explanations of concepts before introducing code.
4. Ensure all code examples are complete, correct, and follow OpenAI API best practices.
5. Add detailed comments in code to explain what each part does.
6. Include proper error handling and API key management.
7. Format the output as a series of Jupyter notebook cells, clearly indicating which are markdown and which are code.
8. Reference official OpenAI documentation where appropriate.
9. Maintain consistency with previously generated content if provided.

### **Output Format:**
For each cell in the notebook section, use the following format:

```markdown
# Your markdown content here
```

```python
# Your Python code here
```

Output:
"""

WRITER_SUBSECTION_PROMPT = """
### **Task:**  
Generate the content for a specific subsection of a Jupyter notebook about OpenAI APIs.

### **Notebook Information:**
- **Title:** {notebook_title}
- **Description:** {notebook_description}
- **Purpose:** {notebook_purpose}
- **Target Audience:** {notebook_target_audience}

### **Section Context:**
- **Parent Section Title:** {parent_section_title}
- **Parent Section Description:** {parent_section_description}

### **Subsection to Generate:**
- **Subsection Title:** {subsection_title}
- **Subsection Description:** {subsection_description}

### **Additional Requirements:**
{additional_requirements}

### **Search Results:**
{search_results}

### **Previously Generated Content:**
{previous_content}

### **Instructions:**
1. Generate both markdown text and Python code cells for this subsection.
2. Follow the subsection description precisely.
3. Include clear explanations of concepts before introducing code.
4. Ensure all code examples are complete, correct, and follow OpenAI API best practices.
5. Add detailed comments in code to explain what each part does.
6. Include proper error handling and API key management.
7. Format the output as a series of Jupyter notebook cells, clearly indicating which are markdown and which are code.
8. Reference official OpenAI documentation where appropriate.
9. Maintain consistency with previously generated content if provided.

### **Output Format:**
For each cell in the notebook subsection, use the following format:

```markdown
# Your markdown content here
```

```python
# Your Python code here
```

Output:
"""

WRITER_REVISION_PROMPT = """
### **Task:**  
Revise the content for a section of a Jupyter notebook about OpenAI APIs based on the evaluation feedback.

### **Notebook Information:**
- **Title:** {notebook_title}
- **Description:** {notebook_description}
- **Purpose:** {notebook_purpose}
- **Target Audience:** {notebook_target_audience}

### **Section Information:**
- **Section Title:** {section_title}
- **Section Description:** {section_description}

### **Original Content:**
{original_content}

### **Evaluation Feedback:**
{evaluation_feedback}

### **Revision Instructions:**
{revision_instructions}

### **Instructions:**
1. Revise the content based on the evaluation feedback.
2. Maintain the same structure and format as the original content.
3. Ensure all code examples are complete, correct, and follow best practices.
4. Format the output as a series of Jupyter notebook cells, clearly indicating which are markdown and which are code.
5. Address all issues mentioned in the evaluation feedback.
6. Implement the suggestions provided in the evaluation feedback.

### **Output Format:**
For each cell in the notebook section, use the following format:

```markdown
# Your markdown content here
```

```python
# Your Python code here
```

Output:
"""
