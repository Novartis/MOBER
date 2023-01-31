#!/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import setup, find_packages

setup(
    name='mober',
    version='2.0.0',
    url='ssh://git@bitbucket.prd.nibr.novartis.net/ods/ods-mober.git',
    author='mober team',
    author_email='gang-6.li@novartis.com',
    description='mober',
    packages=find_packages(),    
    install_requires=['mlflow', 'scanpy'],
    
    keywords='mober',

    entry_points={
        'console_scripts': ['mober = mober.mober:main']
    }
)