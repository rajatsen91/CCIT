from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()

setup(name='CCIT',
      version='0.5',
      description='Model Powered CI Test',
      url='https://github.com/rajatsen91/CCIT.git',
      author='Rajat Sen',
      classifiers=[
        'Development Status :: 5 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      author_email='rsen91@gmail.com',
      license='Apache License 2.0',
      packages=['CCIT'],
      install_requires=[
          'markdown',
          'xgboost==1.5.2',
          'pandas',
          'numpy==1.18.5',
          'scikit-learn==0.23.1',
          'scipy==1.4.1',
          'matplotlib'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
