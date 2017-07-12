from setuptools import setup

setup(name='CCIT',
      version='0.1',
      description='Model Powered CI Test',
      url='https://github.com/rajatsen91/CCIT.git',
      author='Rajat Sen',
      classifiers=[
        'Development Status :: Alpha',
        'License :: Apache License 2.0',
        'Programming Language :: Python :: 2.7',
        'Topic :: Statistics/Machine Learning/Conditional Independence Testing',
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
