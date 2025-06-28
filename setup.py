from setuptools import setup, find_packages

setup(
    name="framesource",
    version="0.1.0",
    description="A flexible, extensible Python framework for acquiring frames from cameras, video files, and image folders.",
    author="Oliver Hamilton",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy",
        "mss"
    ],
    python_requires=">=3.7",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
