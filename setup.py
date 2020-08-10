from setuptools import setup

setup(name='torch-rgvae',
      version='0.1',
      description='A PyTorch library for Relational Graph Variational Auto-Encoder',
      url='https://github.com/INDElab/rgvae',
      author='Thiviyan Thanapalasingam',
      author_email='t.singam@uva.nl',
      license='MIT',
      packages=['torch_rgvae'],
      python_requires='>=3.8',
      zip_safe=False)