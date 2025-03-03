#!/bin/bash
# Installation script for the Cookbook Agent package

echo "Installing Cookbook Agent in development mode..."

# Make script executable
chmod +x install.sh

# Install the package in development mode
pip install -e .

echo "Installation complete! You can now run the main.py script from anywhere."
echo "Try running: python main.py" 