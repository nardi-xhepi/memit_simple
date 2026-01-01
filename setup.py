from setuptools import setup, find_packages

setup(
    name="memit_simple",
    version="0.5.0",
    description="A simple implementation of MEMIT (Mass-Editing Memory in a Transformer)",
    author="Nardi Xhepi",
    packages=["memit_simple"],
    package_dir={"memit_simple": "."},
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
        "datasets",
        "accelerate",
        "mistral-common",
        "transformers @ git+https://github.com/huggingface/transformers@a7f29523361b2cc12e51c1f5133d95f122f6f45c",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
