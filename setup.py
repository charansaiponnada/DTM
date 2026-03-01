from setuptools import setup, find_packages

setup(
    name="dtm-drainage-ai",
    version="1.0.0",
    description="AI/ML DTM Generation and Drainage Network Design for MoPR Hackathon",
    packages=find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "dtm-pipeline=pipelines.full_pipeline:main",
        ],
    },
)
