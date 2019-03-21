from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()

setup(name='CCIT',
      version='0.3',
      description='Model Powered CI Test',
      url='https://github.com/rajatsen91/CCIT.git',
      author='Rajat Sen',
      classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      author_email='rsen91@gmail.com',
      license='Apache License 2.0',
      packages=['CCIT'],
      install_requires=[
          'markdown',
          'xgboost',
          'pandas',
          'numpy',
          'scikit-learn',
          'scipy',
          'matplotlib'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
