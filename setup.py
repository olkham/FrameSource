from setuptools import setup, find_packages
import os

# Dynamically read version from frame_source/_version.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "frame_source", "_version.py")
    about = {}
    with open(version_file) as f:
        exec(f.read(), about)
    return about["__version__"]

setup(
    name="framesource",
    version=get_version(),
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
