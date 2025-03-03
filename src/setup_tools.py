"""
Setup script to make the tools package available in Python path.

This script adds the tools package to the PYTHONPATH so that it can be imported
without needing to modify sys.path in each script.
"""

import os
import sys
import site


def setup_tools_package():
    """
    Add the tools package to the Python path.

    This makes the tools package available for import without needing to modify
    sys.path in each script.
    """
    # Get the path to the current directory (src)
    src_dir = os.path.dirname(os.path.abspath(__file__))

    # Create a .pth file in the site-packages directory
    pth_file = os.path.join(site.getsitepackages()[0], "cookbook_tools.pth")

    with open(pth_file, "w") as f:
        f.write(src_dir)

    print(f"Tools package added to Python path at: {pth_file}")
    print(f"You can now import from 'tools' directly in your scripts.")


if __name__ == "__main__":
    setup_tools_package()
