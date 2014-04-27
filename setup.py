from distutils.core import setup

setup(
    name='graph-tool-nn',
    version='0.0.1',
    packages=['gtnn', ],
    license='GNU General Public License v3 (GPLv3)',
    long_description=open('README.md').read(),

    author="Jan Olszak",
    author_email="janekolszak@gmail.com",
    description="Artificial neural networks implementation using graph-tool",
    keywords="graph-tool neural networks",
    url="http://github.com/janekolszak/gtnn",

    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
)
