# Cookbook Agent

AI Agent for generating educational Jupyter notebooks about OpenAI APIs.

## Features

- **Interactive Clarification**: Uses tools to ask clarification questions when information is ambiguous or missing
- **Structured Outputs**: Utilizes OpenAI's beta.chat.completions.parse for type-safe structured outputs
- **Robust Error Handling**: Provides fallbacks and graceful error reporting
- **Modular Tools**: Organized reusable tools in a dedicated `tools/` package

## Installation

You can install the package in development mode:

```bash
# Clone the repository
git clone https://github.com/yourusername/cookbook_agent.git
cd cookbook_agent

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
./install.sh
# Or manually: pip install -e .
```

## Usage

You can run the main application from the command line:

```bash
python main.py
```

The application will:
1. Prompt you for notebook requirements
2. Ask clarification questions if needed
3. Generate a structured notebook plan
4. Save the plan to notebook_plan.md

## LangSmith Tracing

This project uses LangSmith for tracing and monitoring. To enable tracing:

1. Create a LangSmith account at https://smith.langchain.com/
2. Copy your API key from the LangSmith dashboard
3. Set up environment variables:
   ```bash
   export LANGSMITH_TRACING=true
   export LANGSMITH_API_KEY=your_langsmith_api_key_here
   export LANGSMITH_PROJECT=your_langsmith_project_name_here
   ```
   Or add them to your `.env` file (see `.env.example`)

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


## Project Structure

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

## Development

Run the test script to verify functionality:

```bash
python src/test_planner.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 