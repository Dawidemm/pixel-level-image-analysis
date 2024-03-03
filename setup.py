from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = [line.strip() for line in f.readlines()]

info = {
    "name": "pixel_level_image_analysis",
    "version": "0.0.1",
    "maintainer": "Dawid Mazur",
    "maintainer_email": "dawidmazur@student.agh.edu.pl",
    "url": 'https://github.com/Dawidemm/pixel-level-image-analysis',
    "packages": find_packages(),
    "description": "RBM for pixel-level multispectral image segmentation",
    "long_description": open('README.md').read(),
    "long_description_content_type": "text/markdown",
    "install_requires": requirements,
    "include_package_data": True,
}

classifiers = [
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Engineering"
]

setup(classifiers=classifiers, **(info))