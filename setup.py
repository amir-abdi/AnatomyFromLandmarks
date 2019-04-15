from setuptools import setup, find_packages

__version__ = '1.0.0'
url = 'https://github.com/amir-abdi/LandmarksToShape'

install_requires = [
    'numpy',
    'wandb',
    'matplotlib',
    'pillow',
    'torch',
    'torchvision',
    'scipy'
]

setup(
    name='LandmarksToShape',
    version=__version__,
    description='Shape generator from surface landmarks',
    long_description=open('README.md').read(),
    author='Amir H. Abdi',
    author_email='amirabdi@ece.ubc.ca',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[
        'pytorch', 'shape-generator', 'surface-mesh', 'mesh',
        'neural-networks', 'landmark', 'deep-generative-models'
    ],
    install_requires=install_requires,
    packages=['src'],
    license='GNU General Public License v3.0')
