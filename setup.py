from setuptools import setup, find_packages

setup(
    name="dtm-drainage-ai",
    version="1.0.1",
    description="AI/ML DTM Generation and Drainage Network Design for MoPR Hackathon",
    packages=find_packages(include=["src", "src.*", "pipelines", "pipelines.*"]),
    python_requires=">=3.10",
    install_requires=[
        "laspy",
        "numpy",
        "scipy",
        "rasterio",
        "rio-cogeo",
        "geopandas",
        "pysheds",
        "shapely",
        "pyproj",
        "fiona",
        "scikit-learn",
        "xgboost",
        "pandas",
        "networkx",
        "pyyaml",
        "click",
        "loguru",
        "rich",
        "tqdm",
        "joblib",
        "opencv-python-headless",
    ],
    entry_points={
        "console_scripts": [
            "dtm-pipeline=run_pipeline:main",
        ],
    },
)
