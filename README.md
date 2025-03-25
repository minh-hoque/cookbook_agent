# Cookbook Agent

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

An AI-powered agent system for generating high-quality educational Jupyter notebooks about OpenAI APIs and AI development concepts.

## 📖 Overview

Cookbook Agent streamlines the creation of educational content through an AI-driven workflow:

1. **Planning**: Generates structured notebook outlines based on user requirements
2. **Writing**: Creates comprehensive content with code examples for each section
3. **Review & Revision**: Automatically reviews and improves content quality

Perfect for technical educators, API documentation teams, and developers learning to integrate AI capabilities into applications.

## ✨ Features

- **Interactive Clarification**: Uses tools to ask clarification questions when information is ambiguous or missing
- **Structured Outputs**: Utilizes OpenAI's beta.chat.completions.parse for type-safe structured outputs
- **Robust Error Handling**: Provides fallbacks and graceful error reporting
- **Modular Tools**: Organized reusable tools in a dedicated `tools/` package
- **Multi-format Export**: Generates content in Markdown, Jupyter Notebook, and JSON formats
- **Auto-critique & Revision**: Self-reviews generated content for accuracy and completeness

## 🔧 Prerequisites

- Python 3.9+
- OpenAI API key
- Git (for cloning the repository)
- Virtual environment tool (venv, conda, etc.)

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/cookbook_agent.git
cd cookbook_agent
```

### 2. Set up a virtual environment

```bash
# Using venv (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n cookbook_agent python=3.9
conda activate cookbook_agent
```

### 3. Install the package

```bash
# Using the installation script
./install.sh

# OR manually
pip install -e .
```

### 4. Set up environment variables

Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY=your_openai_api_key

# Optional: For LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=cookbook_agent
```

## 🖥️ Usage

### Basic Usage

Run the main application to generate a notebook plan:

```bash
python main.py
```

This will:
1. Prompt you for notebook requirements
2. Ask clarification questions if needed
3. Generate a structured notebook plan
4. Save the plan to `notebook_plan.md`

### Generate Notebook Content from Plan

Once you have a plan, use the writer module to generate content:

```bash
python -m src.tests.test_writer --plan notebook_plan.md --output ./output
```

Options:
- `--model <model_name>`: Specify the OpenAI model to use (default: gpt-4o)
- `--section <index>`: Generate content for a specific section only (0-based index)
- `--parse-only`: Only parse the plan without generating content

Example:
```bash
python -m src.tests.test_writer --model gpt-4o --plan ./data/agents_sdk_plan.md
```

### Output Files

Generated content is saved to the specified output directory (default: `./output`):
- `notebook_plan.json`: JSON representation of the parsed plan
- `notebook_<title>.md`: Markdown version of the complete notebook
- `notebook_<title>.ipynb`: Jupyter Notebook version
- `section_<n>_<title>.json`: Individual section content

## 📊 LangSmith Tracing

This project uses LangSmith for tracing and monitoring. To enable tracing:

1. Create a LangSmith account at https://smith.langchain.com/
2. Copy your API key from the LangSmith dashboard
3. Set up environment variables:
   ```bash
   export LANGSMITH_TRACING=true
   export LANGSMITH_API_KEY=your_langsmith_api_key_here
   export LANGSMITH_PROJECT=your_langsmith_project_name_here
   ```
   Or add them to your `.env` file

### With LangChain Components

If you're using LangChain components, tracing is automatically enabled when the environment variables are set.

### With LangGraph

For LangGraph workflows:

```python
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI

# LangSmith will automatically trace LangChain components
model = ChatOpenAI()

# Create your graph
workflow = StateGraph(MessagesState)
# Add nodes and edges...
app = workflow.compile()

# Invoke with optional thread_id for grouping related traces
final_state = app.invoke(
    {"messages": [HumanMessage(content="your query")]},
    config={"configurable": {"thread_id": "unique_id"}}
)
```

## 📁 Project Structure

```
cookbook_agent/
├── main.py                 # Main entry point
├── test_main.py            # Test for main application
├── setup.py                # Package setup
├── install.sh              # Installation script
├── requirements.txt        # Project dependencies
├── .env.example            # Example environment variables
├── notebook_plan.md        # Generated notebook plan
├── test_notebook_plan.md   # Test notebook plan
├── notebook_generator_flow.md # Flow documentation
├── data/                   # Data directory
├── output/                 # Output directory
├── logs/                   # Log files
└── src/                    # Source code
    ├── __init__.py         # Package initialization
    ├── models.py           # Pydantic models
    ├── planner.py          # PlannerLLM implementation
    ├── searcher.py         # Search functionality
    ├── writer.py           # Notebook writer
    ├── user_input.py       # User input handling
    ├── format/             # Formatting utilities
    │   ├── __init__.py
    │   ├── format_utils.py
    │   ├── examples.py
    │   ├── convert_multiple_files.py
    │   └── json_to_md_example.py
    ├── prompts/            # Prompt templates
    │   ├── __init__.py
    │   ├── planner_prompts.py
    │   ├── writer_prompts.py
    │   └── critic_prompts.py
    ├── tools/              # LLM tools
    │   ├── clarification_tools.py
    │   ├── debug.py
    │   └── utils.py
    └── tests/              # Test modules
        ├── __init__.py
        ├── test_planner.py
        ├── test_searcher.py
        └── test_writer.py
```

## 🧪 Development

### Running Tests

Verify functionality with the test scripts:

```bash
# Test the planner module
python -m src.tests.test_planner

# Test the writer module
python -m src.tests.test_writer --parse-only
```

### Common Issues

- **"OpenAI API key not found"**: Set your API key in the `.env` file or as an environment variable
- **Import errors**: Make sure you've activated your virtual environment and installed with `pip install -e .`
- **Library not found errors**: Check that all dependencies were installed with `pip install -r requirements.txt`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 