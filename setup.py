from setuptools import setup

setup(
    name='set50predictor',
    version='0.1.0',
    description='Utilities function for predict SET50 stock price using LSTM',
    url='https://github.com/pathompong-y/set50predictor',
    author='Pathompong Yupensuk',
    author_email='pathompong.y@gmail.com',
    license='MIT',
    packages=['set50predictor'],
    install_requires=['yfinance',
                      'numpy',
                      'pandas',
                      'sklearn',
                      'matplotlib',
                      'keras',
                      'tensorflow'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
    ],
)