"""
Setup script for the Cookbook Agent package.

This is a minimal setup script to allow importing the cookbook_agent package.
"""

from setuptools import setup, find_packages

setup(
    name="cookbook_agent",
    version="0.1.0",
    description="AI Agent for Generating OpenAI Demo Notebooks",
    author="Minhajul",
    packages=find_packages(),
    install_requires=[
        "openai>=1.10.0",
        "pydantic>=2.0.0",
    ],
    python_requires=">=3.8",
)
