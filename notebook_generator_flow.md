```mermaid
flowchart TD
    A((User))
    A -->|1-Provide Notebook Description & Optional Requirements Snippets| B[Planner LLM]
    B --> C{2- Needs Clarification?}
    C --|Yes: Ask Questions|--> A
    C --|No: Produce Detailed Outline|--> D[Outline]
    D -->|3- Send Outline| E[Writer LLM]
    E -->|4- Generate Section Content| F[Critic LLM]
    F --> G{5- Issues Found?}
    G --|Yes: Provide Revision Feedback|--> E
    G --|No: All Sections Approved|--> H[Format LLM]
    H -->|6- Assemble Final Notebook| I[[.ipynb Output]]
```