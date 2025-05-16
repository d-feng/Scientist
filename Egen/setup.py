from setuptools import setup, find_packages

setup(
    name="e_gen",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "pydantic",
        "langchain-core",
        "langgraph",
        "langchain-openai",
        "langchain-anthropic",
        "python-dotenv"
    ],
    extras_require={
        "dev": ["pytest"]
    },
    python_requires=">=3.8",
    author="Your Name",
    description="E-Gen framework for hypothesis testing and data analysis",
    keywords="hypothesis-testing, data-analysis, eQTL",
) 