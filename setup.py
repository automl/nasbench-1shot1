import os
from setuptools import setup, find_packages

VERBOSE_SCRIPT = True

cwd = os.path.dirname(os.path.abspath(__file__))
version = open('.version', 'r').read().strip()

if VERBOSE_SCRIPT:
    def report(*args):
        print(*args)
else:
    def report(*args):
        pass

version_path = os.path.join(cwd, 'nasbench1shot1', '__init__.py')
with open(version_path, 'w') as f:
    report('-- Building version ' + version)
    f.write("__version__ = '{}'\n".format(version))


if __name__=='__main__':
    setup(
        name='nasbench1shot1',
        version=version,
        description='NAS-Bench-1Shot1',
        author='Julien Siems, Arber Zela',
        author_email='zelaa@cs.uni-freiburg.de',
        url='https://github.com/automl/nasbench-1shot1',
        license='Apache License 2.0',
        classifiers=['Development Status :: 1 - Beta'],
        packages=find_packages(),
        python_requires='>=3',
        install_requires=['ConfigSpace'],
        keywords=['benchmark', 'NAS', 'automl'],
        #test_suite='tests'
    )
