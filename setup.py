from setuptools import setup, find_packages

setup(
    name="fft_tdse",
    author="Simen Kvaal",
    version="0.1",
    description="",
    long_description="",
    packages=find_packages(),
    install_requires=[
        "scipy", "hsluv", "colorcet",
#        "repos @ git+https://github.com/.../repos",
    ],
#    test_suite="tests", 
    entry_points={
        'console_scripts': ['simulate2d=scripts.simulate2d:main']
    }   
)
