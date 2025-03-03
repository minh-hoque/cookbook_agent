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

## Project Structure

```
cookbook_agent/
├── main.py                 # Main entry point
├── setup.py                # Package setup
├── install.sh              # Installation script
└── src/                    # Source code
    ├── models.py           # Pydantic models
    ├── planner.py          # PlannerLLM implementation
    ├── prompts/            # Prompt templates
    │   └── planner_prompts.py
    ├── tools/              # LLM tools
    │   ├── __init__.py
    │   ├── clarification_tools.py
    │   └── utils.py
    └── test_planner.py     # Test script
```

## Development

Run the test script to verify functionality:

```bash
python src/test_planner.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 