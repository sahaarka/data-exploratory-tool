from setuptools import setup, find_packages

setup(
    name="data-discovery-tool",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit>=1.22.0",
        "pandas>=1.5.0",
        "numpy>=1.22.0",
        "plotly>=5.10.0",
        "seaborn>=0.12.0",
        "matplotlib>=3.6.0",
        "scikit-learn>=1.1.0",
    ],
    entry_points={
        "console_scripts": [
            "data-discovery-tool=data_discovery_tool.cli:main",
        ],
    },
    python_requires=">=3.8",
    author="Arka Saha",
    author_email="anticlock909@gmail.com",
    description="Advanced Data Discovery Tool",
    keywords="data, analysis, discovery, streamlit",
)
