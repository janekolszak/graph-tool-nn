from distutils.core import setup

setup(
    name='graph-tool-nn',
    version='1.0.2',
    packages=['gtnn', 'gtnn.generators', 'gtnn.learn', 'gtnn.network'],
    license='GNU General Public License v3 (GPLv3)',
    long_description=r'''Graph-tool is a great open source tool for creating, using and analyzing graphs. It's a python
library with C++ bindings, uses boost::graph under the hood and seems to be pretty fast
(http://graph-tool.skewed.de/).

Graph-tool Neural Networks (gtnn) is an implementation of ANN on top of graph-tool. It makes
researching neural networks nice&easy. You can create custom nets, train, analyze and plot them.

''',

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
