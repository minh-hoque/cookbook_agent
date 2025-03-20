"""
Prompt templates for the Writer LLM.

This module contains the prompt templates used by the Writer LLM to generate
notebook content based on the plan created by the Planner LLM.
"""

# Web search prompt for detailed API information with code examples
WEB_SEARCH_PROMPT = """
When searching for information, provide detailed, clear, and comprehensive answers that directly address the query. 

Focus on:
1. Providing accurate and up-to-date information relevant to the query
2. Including practical code examples in Python whenever the topic involves programming or APIs
4. Including version information and noting any recent changes or deprecations for software-related topics
5. Organizing information in a structured, easy-to-understand format
6. Citing official documentation and reliable sources

When searching for information about OpenAI APIs, provide code, examples, and explanations based on the official OpenAI API documentation.
Do not include information that is not available in the official OpenAI API documentation.

This information will be used to create educational Python notebooks, so clarity, accuracy, and depth are essential.
"""

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

### **Tone and Style:**
- **Friendly and Encouraging:** Write as if you're guiding a student. Keep the tone positive and supportive.
- **Clear and Concise:** Avoid unnecessary complexity; use simple language.
- **Engaging and Practical:** Use real-world examples and interactive exercises to keep users engaged.
- **Well-Commented Code:** Every code cell should have comments explaining what each part does.

### **Additional Considerations:**
- **Ensure the notebook runs smoothly without errors.**
- **Format markdown cells properly for readability (headings, bullet points, bold text).**
- **Use visualizations (matplotlib, seaborn) when applicable to explain data-related concepts.**
"""

WRITER_CONTENT_PROMPT = """
### Task
Generate the content for a section of a Jupyter notebook about OpenAI APIs.

---

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

---

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
    {{
        "cell_type": "markdown",
        "content": "Your markdown content here"
    }},
    {{
        "cell_type": "code",
        "content": "Your Python code here"
    }},
    {{
        "cell_type": "code",
        "content": "Your Python code here"
    }}
]
```

Output:
"""


WRITER_REVISION_PROMPT = """
### **Task:**  
Revise the content for a section of a Jupyter notebook about OpenAI APIs based on the evaluation feedback.

---

### **Notebook Details:**  
- **Title:** {notebook_title}  
- **Description:** {notebook_description}  
- **Purpose:** {notebook_purpose}  
- **Target Audience:** {notebook_target_audience}  

### **Section Details:**  
- **Section Title:** {section_title}  
- **Section Description:** {section_description}  

### **Original Content:**  
{original_content}  

### **Evaluation Feedback:**  
{evaluation_feedback}  

---

### **Revision Instructions:**  

1. **Review the Feedback:**  
   - Carefully analyze the evaluation feedback.  
   - Identify all suggested improvements and issues that need to be addressed.  

2. **Revise the Content Accordingly:**  
   - Implement all necessary changes while keeping the original structure intact.  
   - Improve explanations, fix errors, and enhance clarity where needed.  
   - Ensure all modifications align with the feedback provided.  

3. **Ensure Code Quality:**  
   - Make sure all Python code examples are **correct, complete, and follow OpenAI API best practices**.  
   - Add **inline comments** to explain each step of the code.  
   - Implement **error handling** and ensure **secure API key management**.  

4. **Maintain Proper Formatting:**  
   - Keep the **same structure** as the original content.  
   - Clearly separate **markdown (explanations) and code cells**.  
   - Ensure the content is properly formatted for a Jupyter notebook.  

5. **Confirm That All Issues Are Fixed:**  
   - Double-check that every issue from the feedback is resolved.  
   - Ensure the content is more accurate, clear, and well-structured after revisions.  

6. **Output Format:**  
   - Provide the revised content as a series of **Jupyter notebook cells** in the following format:  

### **Output Format:**
For each cell in the notebook section, use the following JSON format:

```json
[
    {{
        "cell_type": "markdown",
        "content": "Your markdown content here"
    }},
    {{
        "cell_type": "code",
        "content": "Your Python code here"
    }},
    {{
        "cell_type": "code",
        "content": "Your Python code here"
    }}
]
```

Revised Output:
"""

# Prompts for the SearchDecisionNode
SEARCH_DECISION_INITIAL_PROMPT = """
### Task
Determine whether additional searches are needed before generating content for a notebook section on OpenAI APIs.

### Current Date
- **Month/Year:** {current_month_year}

### Notebook Information
- **Title:** {notebook_title}
- **Description:** {notebook_description}
- **Purpose:** {notebook_purpose}
- **Target Audience:** {notebook_target_audience}

### Section to Evaluate
- **Section Title:** {section_title}
- **Section Description:** {section_description}

### Subsections to Include
{subsections_details}

### Additional Requirements
{additional_requirements}

### Existing Search Results
{existing_search_results}

### Previous Content (If Available)
{previous_content}

### Instructions
1. **Analyze the section requirements** to understand what information is needed.
2. **Review the existing search results** to determine if they provide sufficient information.
3. **Decide if additional searches are needed** based on any knowledge gaps identified.
4. If additional searches are needed, generate specific, targeted search queries to fill those gaps.
5. **Remember to consider the current date ({current_month_year})** when formulating queries to ensure you get the most up-to-date information.

### Decision-Making Criteria
- Are the existing search results comprehensive enough to cover all subsections?
- Do the results provide up-to-date and accurate information on OpenAI API features?
- Are there any technical details, parameters, or concepts missing that would be essential for coding examples?
- Would additional searches provide valuable information not currently available?
- Is there time-sensitive information related to API changes or updates that needs to be considered?

Please respond with a structured decision and justification.
"""

SEARCH_DECISION_POST_CRITIQUE_PROMPT = """
### Task
Re-evaluate whether additional searches are needed after receiving critique on the generated content.

### Current Date
- **Month/Year:** {current_month_year}

### Notebook Information
- **Title:** {notebook_title}
- **Description:** {notebook_description}
- **Purpose:** {notebook_purpose}
- **Target Audience:** {notebook_target_audience}

### Section to Evaluate
- **Section Title:** {section_title}
- **Section Description:** {section_description}

### Subsections to Include
{subsections_details}

### Additional Requirements
{additional_requirements}

### Existing Search Results
{existing_search_results}

### Generated Content
{generated_content}

### Critic Evaluation
{critic_evaluation}

### Instructions
1. **Analyze the critic's feedback** to identify specific information gaps or errors in the content.
2. **Review the existing search results** to determine if they adequately address these issues.
3. **Decide if additional targeted searches are needed** to resolve the specific problems identified by the critic.
4. If additional searches are needed, generate specific, targeted search queries to address the critic's concerns.
5. **Consider the current date ({current_month_year})** to ensure search queries will return the most recent, relevant information.

### Decision-Making Criteria
- Does the critic's feedback point to missing information that could be filled with additional searches?
- Are there factual errors or outdated information that needs correction through fresh search results?
- Are there technical aspects or code examples that need more comprehensive or accurate information?
- Would additional searches significantly improve the content quality based on the critic's feedback?
- Are there recent changes to the API (as of {current_month_year}) that should be addressed?

Please respond with a structured decision and justification, focusing on how additional searches would address the specific issues raised by the critic.
"""
