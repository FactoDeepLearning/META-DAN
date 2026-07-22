from pathlib import Path
from setuptools import find_packages, setup

setup(
    name="dar",
    version="0.1",
    description="Package for document analysis and recognition (dar)",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Denis Coquenet",
    author_email="denis.coquenet@irisa.fr",
    python_requires=">=3.9",
    install_requires=[
        "torch==2.4.0",
        "torchvision==0.19",
        "datasets==3.0.0",
        "tokenizers==0.20.0",
        "sentencepiece==0.2.0",
        "timm==1.0.9",
        "pyyaml==6.0.2",
        "pillow==9.5.0",
        "numpy==1.26.4",
        "scipy==1.11.4",
        "matplotlib==3.9.2",
        "tensorboard==2.17.1",
        "tqdm==4.66.5",
        "editdistance==0.8.1",
        "lxml==5.3.0",
        "opencv-python==4.10.0.84",
        "zipfile_deflate64==0.2.0",
        "wget==3.2",
        "python-bidi==0.6.0",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "dar=dar.cli:main",
        ]
    },
)
