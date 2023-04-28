import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tensorgrad",
    version="0.1.0",
    author="Kexin Hu",
    author_email="hukexin0000@126.com",
    description="A tiny tensor-valued autograd engine with a small PyTorch-like neural network library on top.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hkxIron/tensorgrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
