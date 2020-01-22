from setuptools import setup, find_packages

setup(
    name='nasbench1shot1',
    version='0.0.1',
    description='NAS-Bench-1Sho1',
    author='Julien Siems, Arber Zela',
    author_email='siemsj@cs.uni-freiburg.de',
    url='https://github.com/automl/nasbench-1shot1',
    license='Apache License 2.0',
    classifiers=['Development Status :: 1 - Beta'],
    packages=find_packages(),
    python_requires='>=3',
    install_requires=['numpy'],
    keywords=['benchmark', 'NAS', 'automl'],
    #test_suite='tests'
)
