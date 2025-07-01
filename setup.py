from setuptools import setup, find_packages

setup(
    name="document-intelligence-system",
    version="1.0.0",
    description="Advanced Agentic Document Intelligence System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Assessment Project",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
    entry_points={
        "console_scripts": [
            "doc-intelligence=main:main",
            "doc-process=interfaces.cli_processor:main",
            "doc-chat=interfaces.cli_chatbot:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)