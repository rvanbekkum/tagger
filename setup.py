# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='Tagger',
    version='0.0.1',
    description='Recommending tags for photos.',
    long_description=readme,
    author='Rob van Bekkum, Aaron Ang',
    author_email='r.vanbekkum@student.tudelft.nl, a.w.z.ang@student.tudelft.nl',
    url='https://github.com/rvanbekkum/tagger',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
