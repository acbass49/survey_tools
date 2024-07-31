from setuptools import setup, find_packages

VERSION = '1.0.0' 
DESCRIPTION = 'Tools for Survey Wrangling and Analysis'
LONG_DESCRIPTION = 'Tools for Survey Wrangling and Analysis for Users of Pandas'

# Setting up
setup(
    
    name="survey_tools", 
    version=VERSION,
    author="Alex Bass",
    author_email="acbass49@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'functools',
        'operator',
        'warnings'
    ],
    
    keywords=['surveys', 'recode', 'weighted crosstabs'],
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)

#following tutorial at: https://www.freecodecamp.org/news/build-your-first-python-package/
