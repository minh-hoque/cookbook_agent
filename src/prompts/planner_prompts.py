"""
Prompt templates for the Planner LLM.

This module contains the prompt templates used by the Planner LLM to generate
clarification questions and notebook plans.
"""

PLANNING_SYSTEM_PROMPT = """
You are an AI assistant that specializes in planning educational Jupyter notebooks about OpenAI APIs.

Your task is to create a **detailed and well-structured outline** for a notebook based on the user's requirements.  
---
### General Guidelines:
- Your output should be a **notebook outline**, not actual content or code.
- Structure the notebook into **5–8 main sections**, each with **relevant subsections**.
- Give every section and subsection a **clear, descriptive title**.
- Provide a **detailed explanation** for each section, describing:
  - What the section covers
  - Why it matters
  - How it supports the notebook’s goals
- Follow a **logical learning path**, starting from core concepts and building up to advanced applications.
- Include sections for:
  - Introduction
  - Setup (briefly!)
  - Key concepts
  - Demonstrations or use cases
  - Summary and/or next steps
- **Do NOT include any reflection questions or exercises at the end of key sections.**
- **Do NOT include any code or implementation details.** This is strictly a planning task.
---
### Important on Setup Section:
The **Setup & Prerequisites** section should be **no more than 1–2 short paragraphs**.  
Do **not include detailed installation or configuration steps** for Python, virtual environments, or the OpenAI SDK.  
Assume the audience has a technical background and can look up common setup instructions.  
Only mention what's needed at a high level in a bulleted list:
- Python 3.12+
- A virtual environment (e.g. `conda` or `venv`)
- Access to the OpenAI API key
---
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
1. **Do NOT write any code or implementation details.**
2. Break the notebook into **5–8 main sections**, each with clear **subsections**.
3. For each section, include a **detailed description** explaining:
   - What the section covers
   - Why the topic is important
   - How it supports the learning goals
4. Follow a **logical learning progression** — start from foundational ideas and move to advanced use cases.
5. You **must** include a **brief Setup & Prerequisites** section, but keep it **to 1–2 short paragraphs** maximum. Do **not** explain how to install Python, create environments, or set up API keys in detail.
6. Include sections such as:
   - Notebook introduction and learning objectives
   - Setup & prerequisites (briefly!)
   - Key concepts with explanations
   - Practical demonstrations (describe, don’t implement)
   - Real-world applications
   - Summary and key takeaways

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
1. **Do NOT write any code or implementation details.**
2. Use the clarifications provided by the user to tailor the notebook outline to the user's needs.
3. Break the notebook into **5–8 main sections**, each with clear **subsections**.
4. For each section, include a **detailed description** explaining:
   - What the section covers
   - Why the topic is important
   - How it supports the learning goals
5. Follow a **logical learning progression** — start from foundational ideas and move to advanced use cases.
6. You **must** include a **brief Setup & Prerequisites** section, but keep it **to 1–2 short paragraphs** maximum. Do **not** explain how to install Python, create environments, or set up API keys in detail.
7. Include sections such as:
   - Notebook introduction and learning objectives
   - Setup & prerequisites (briefly!)
   - Key concepts with explanations
   - Practical demonstrations (describe, don’t implement)
   - Real-world applications
   - Summary and key takeaways

### **Expected Output Format:**  
A structured outline that clearly defines the notebook's sections, their content, and the key points to be covered.  

Output:
"""
