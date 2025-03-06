"""
Prompt templates for the Planner LLM.

This module contains the prompt templates used by the Planner LLM to generate
clarification questions and notebook plans.
"""

PLANNING_SYSTEM_PROMPT = """
You are an AI assistant that specializes in planning educational Jupyter notebooks about OpenAI APIs.

Your task is to create a **detailed and well-structured outline** for a notebook based on the user's requirements.  

### **Key Guidelines:**
- Create a notebook outline with **5-8 main sections**, each with subsections.
- Give each section and subsection a **clear, descriptive title** that conveys its specific content.
- For each section, include a **detailed explanation** of what it covers and its educational value.
- Design a **logical learning progression** that starts with fundamentals and builds to more complex concepts.
- Include explicit sections for setup, introduction to concepts, practical examples, and conclusion.
- **Do not write any actual code**—this is strictly an outline for the notebook structure.
- Consider including reflection questions or exercises at the end of key sections.

### **Tools:**
- `get_clarifications`: Use this tool to get clarification from the user if needed. For example, if the user's requirements are not clear, you can use this tool to get clarification.
"""

PLANNING_PROMPT = """
### **Task:**  
Create a **detailed outline** for an educational Jupyter notebook about OpenAI APIs based on the following specifications.

### **Notebook Details:**  
- **Description:** {description}  
- **Purpose:** {purpose}  
- **Target Audience:** {target_audience}  
- **Additional Requirements:** {additional_requirements}  
- **Code Snippets (if applicable):** {code_snippets}

### **Search Results:**
{search_results}

### **Instructions:**  
1. **Do not write any code or implementation details** - only create a structured outline.
2. Create **5-8 main sections** with **subsections** each, all with descriptive titles.
3. For each section, write a **detailed description** explaining:
   - What topics it covers
   - Why these topics are important
   - How they connect to the notebook's purpose
4. Ensure the outline follows a **logical educational progression** from basic to advanced concepts.
5. Include specific sections for:
   - Setup and prerequisites
   - Concept introduction
   - Step-by-step demonstrations (without actual code)
   - Practical applications
   - Summary and next steps

### **Expected Output Format:**  
A structured outline that clearly defines the notebook's sections, their content, and the key points to be covered.  

Output:
"""

PLANNING_WITH_CLARIFICATION_PROMPT = """
### **Task:**  
Create a **detailed outline** for an educational Jupyter notebook based on the following topic and requirements. This outline will serve as the foundation for generating the actual notebook.

### **Notebook Details:**  
- **Description:** {description}  
- **Purpose:** {purpose}  
- **Target Audience:** {target_audience}  
- **Additional Requirements:** {additional_requirements}  
- **Code Snippets (if applicable):** {code_snippets}

### **Clarifications:**
{clarifications}

### **Search Results:**
{search_results}

### **Instructions:**  
1. **Do not write any code or implementation details** - only create a structured outline.
2. Create **5-8 main sections** with **subsections** each, all with descriptive titles.
3. For each section, write a **detailed description** explaining:
   - What topics it covers
   - Why these topics are important
   - How they connect to the notebook's purpose
4. Ensure the outline follows a **logical educational progression** from basic to advanced concepts.
5. Include specific sections for:
   - Setup and prerequisites
   - Concept introduction
   - Step-by-step demonstrations (without actual code)
   - Practical applications
   - Summary and next steps

### **Expected Output Format:**  
A structured outline that clearly defines the notebook's sections, their content, and the key points to be covered.  

Output:
"""


def format_additional_requirements(requirements):
    """Format additional requirements for the prompt."""
    if not requirements:
        return "None specified"

    formatted = ""

    # Handle both list and dict inputs
    if isinstance(requirements, list):
        for req in requirements:
            formatted += f"- {req}\n"
    elif isinstance(requirements, dict):
        for key, value in requirements.items():
            if key not in [
                "description",
                "purpose",
                "target_audience",
                "code_snippets",
            ]:
                formatted += f"- {key}: {value}\n"
    else:
        formatted = f"- {str(requirements)}\n"

    return formatted.strip() or "None specified"


def format_code_snippets(snippets):
    """Format code snippets for the prompt."""
    if not snippets:
        return "None provided"

    formatted = ""
    for i, snippet in enumerate(snippets, 1):
        formatted += f"Snippet {i}:\n```python\n{snippet}\n```\n\n"

    return formatted.strip() or "None provided"


def format_clarifications(clarifications):
    """Format clarifications for the prompt."""
    if not clarifications:
        return "No clarifications provided"

    formatted = ""
    for question, answer in clarifications.items():
        formatted += f"Q: {question}\nA: {answer}\n\n"

    return formatted.strip() or "No clarifications provided"


def format_search_results(search_results):
    """Format search results for the prompt."""
    if not search_results:
        return "No search results available"

    return search_results.strip() or "No search results available"
