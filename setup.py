import os

from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "mzt",
    version = "1.1.4-alpha",
    author = "Shivam Pandey",
    author_email = "pandeyshivam2017robotics@gmail.com",
    description = ("Package to host DeepLearning modules for pytorch ecosystem,"
                   " to ease out model implementations."),
    license = "AGPLv3+",
    keywords = "DeepLearning Pytorch Modules",
    url = "https://github.com/ShivamPR21/ModuleZooTorch.git",
    packages=['moduleZoo',
              'moduleZoo.convolution',
              'moduleZoo.resblocks',
              'moduleZoo.attention',
              'moduleZoo.dense',
              'moduleZoo.graphs',
              'modelZoo',
              'mzLosses',
              'mzExtras'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
    ],
)
