"""
User Input Handler Module

This module is responsible for collecting notebook requirements from the user.
It provides a structured interface to gather information about the notebook's
purpose, audience, and any additional requirements or code snippets.
"""


class UserInputHandler:
    """
    Handles the collection of user input for notebook generation.

    This class provides methods to collect and validate user requirements
    for generating OpenAI API demonstration notebooks.
    """

    def __init__(self):
        """Initialize the UserInputHandler."""
        self.requirements = {
            "notebook_description": "",
            "notebook_purpose": "",
            "target_audience": "",
            "additional_requirements": [],
            "code_snippets": [],
        }

    def collect_requirements(self):
        """
        Collect all necessary requirements from the user.

        Returns:
            dict: A dictionary containing all the collected requirements.
        """
        print("\n=== Notebook Description & Purpose ===")
        self._collect_notebook_description()
        self._collect_notebook_purpose()
        self._collect_target_audience()

        print("\n=== Additional Requirements (Optional) ===")
        self._collect_additional_requirements()

        print("\n=== Code Snippets (Optional) ===")
        self._collect_code_snippets()

        return self.requirements

    def _collect_notebook_description(self):
        """Collect the notebook description from the user."""
        prompt = (
            "Please provide a brief description of the notebook you want to create: "
        )
        self.requirements["notebook_description"] = input(prompt).strip()

        # Keep asking until we get a non-empty description
        while not self.requirements["notebook_description"]:
            print("Notebook description cannot be empty.")
            self.requirements["notebook_description"] = input(prompt).strip()

    def _collect_notebook_purpose(self):
        """Collect the notebook purpose from the user."""
        prompt = "What is the main purpose of this notebook? (e.g., tutorial, demonstration, exploration): "
        self.requirements["notebook_purpose"] = input(prompt).strip()

        # Keep asking until we get a non-empty purpose
        while not self.requirements["notebook_purpose"]:
            print("Notebook purpose cannot be empty.")
            self.requirements["notebook_purpose"] = input(prompt).strip()

    def _collect_target_audience(self):
        """Collect information about the target audience."""
        prompt = "Who is the target audience for this notebook? (e.g., beginners, experienced developers): "
        self.requirements["target_audience"] = input(prompt).strip()

        # This can be optional, so we don't enforce non-empty
        if not self.requirements["target_audience"]:
            self.requirements["target_audience"] = "General users"

    def _collect_additional_requirements(self):
        """Collect any additional requirements from the user."""
        print(
            "Do you have any additional requirements? (e.g., docstring formats, environment settings)"
        )
        print("Enter requirements one by one. Type 'done' when finished.")

        while True:
            requirement = input("Requirement (or 'done' to finish): ").strip()
            if requirement.lower() == "done":
                break
            if requirement:  # Only add non-empty requirements
                self.requirements["additional_requirements"].append(requirement)

    def _collect_code_snippets(self):
        """Collect any code snippets the user wants to include."""
        print("Do you have any specific code snippets you want to include?")
        print("Enter snippets one by one. Type 'done' when finished.")
        print("For multi-line snippets, type 'multiline' and then enter your code.")
        print("When finished with a multi-line snippet, type 'end' on a new line.")

        while True:
            snippet_prompt = input(
                "Snippet (or 'done' to finish, 'multiline' for multi-line code): "
            ).strip()

            if snippet_prompt.lower() == "done":
                break

            if snippet_prompt.lower() == "multiline":
                print(
                    "Enter your multi-line code snippet (type 'end' on a new line when finished):"
                )
                lines = []
                while True:
                    line = input()
                    if line.strip().lower() == "end":
                        break
                    lines.append(line)

                if lines:  # Only add non-empty snippets
                    self.requirements["code_snippets"].append("\n".join(lines))
            elif snippet_prompt:  # Only add non-empty snippets
                self.requirements["code_snippets"].append(snippet_prompt)
