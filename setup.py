from setuptools import setup
from distutils.util import convert_path
from os import path


here = path.abspath(path.dirname(__file__))
version_dict = {}
version_path = convert_path('tf_sprinkles/_version.py')

# Get the version number from the version path.
with open(version_path, 'r') as ver_file:
    exec(ver_file.read(), version_dict)

# Get the long description from the README file.
with open(path.join(here, 'README.md')) as a_file:
    long_description = a_file.read()

CLASSIFIERS = ['Development Status :: 5 - Production/Stable',
               'Environment :: Console',
               'Intended Audience :: Developers',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: MIT License',
               'Natural Language :: English',
               'Operating System :: OS Independent',
               'Programming Language :: Python',
               'Programming Language :: Python :: 3.7',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Artificial Intelligence',
               'Topic :: Utilities']
PROJECT_URLS = {'Documentation': 'https://engineero.github.io/tf_sprinkles',
                'Source': 'https://github.com/Engineero/tf_sprinkles',
                'Tracker': 'https://github.com/Engineero/tf_sprinkles/issues'}
INSTALL_REQUIRES = ['tensorflow>=2.0']

setup(name='tf_sprinkles',
      packages=['tf_sprinkles'],
      version=version_dict['__version__'],
      description='A fast and efficient implimentation of progressive sprinkles augmentation.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Engineero',
      author_email='engineerolabs@gmail.com',
      url='https://github.com/Engineero/tf_sprinkles',
      project_urls=PROJECT_URLS,
      keywords=['augmentation sprinkles tensorflow'],
      classifiers=CLASSIFIERS,
      install_requires=INSTALL_REQUIRES)
