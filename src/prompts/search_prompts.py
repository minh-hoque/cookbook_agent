"""
Prompt templates for searching information.

This module contains prompts used to guide web searches for obtaining information
about OpenAI APIs and other technical topics needed for notebook creation.
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
