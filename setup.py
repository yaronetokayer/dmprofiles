import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='dmprofiles',
    version='0.0.2',
    author='Yarone Tokayer',
    author_email='yarone.tokayer@yale.edu',
    description='A collection of functions for generating and fitting dark matter halo profiles',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/yaronetokayer/dmprofiles',
    project_urls = {
        "Bug Tracker": "https://github.com/yaronetokayer/dmprofiles/issues"
    },
    license='GNU GPLv3',
    packages=[
        'dmprofiles'
    ],
    install_requires=[
        'numpy',
        'astropy',
        'sigfig',
        'scipy'
    ],
)