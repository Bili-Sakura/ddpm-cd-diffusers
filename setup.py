"""Package setup for ddpm-cd-diffusers."""
from setuptools import find_packages, setup

setup(
    name="ddpm-cd-diffusers",
    version="0.1.0",
    description="Change Detection using Diffusion Model Features (HuggingFace diffusers)",
    packages=find_packages(where=".", include=["src", "src.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "diffusers>=0.20.0",
        "transformers>=4.25.0",
        "accelerate>=0.20.0",
        "PyYAML>=6.0",
        "numpy>=1.23.0",
        "tqdm>=4.64.0",
        "opencv-python>=4.6.0",
        "Pillow>=9.0.0",
        "tensorboardX>=2.5.0",
        "wandb>=0.13.0",
    ],
)
