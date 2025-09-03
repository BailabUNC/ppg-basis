from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'jupyterlab',
    'matplotlib',
    'numba',
    'scipy',
    'fastplotlib',
    'ipywidgets',
    'warnings',
    'typing'
]

setup(
    name='ppg_basis',
    version='1.0',
    packages=find_packages(),
    install_requires=install_requires,
    url='',
    license='',
    author='Arjun Putcha',
    author_email='arjunputcha@gmail.com',
    description='Tool for PPG Decomposition using various basis functions'
)