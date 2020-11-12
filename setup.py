from setuptools import setup

setup(
    name="multithread-multicam",
    version="0.1",
    url="https://github.com/nrsyed/multithread-multicam",
    author="Najam R. Syed",
    author_email="najam.r.syed@gmail.com",
    license="MIT",
    packages=["multicam"],
    install_requires=[
        "numpy",
        "opencv-python",
        "screeninfo",
    ],
    entry_points={
        "console_scripts": ["multicam = multicam.__main__:main"],
    },
)
