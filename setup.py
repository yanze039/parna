# -*- coding: utf-8 -*-

from os import path
import setuptools
import datetime

today = datetime.date.today().strftime("%b-%d-%Y")

# read requirements
install_requires = []
with open(path.join(path.abspath(path.dirname(__file__)), 'requirements.txt')) as f:
    for line in f.readlines():
        install_requires.append(line.strip())

setuptools.setup(
    name='parna',
    author="Yanze Wang",
    author_email="yanze039@mit.edu",
    description="PArameterization tool for RNA",
    long_description="RESP and Geometry modeling",
    long_description_content_type="text/markdown",
    url="",
    python_requires=">=3.10",
    packages=['parna'],
    classifiers=[
        "Programming Language :: Python :: 3.10",
    ],
    include_package_data=True,
    install_requires=install_requires,
)