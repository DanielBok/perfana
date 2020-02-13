import re

from setuptools import find_packages, setup

PACKAGE_NAME = 'perfana'

install_requires = [
    'copulae >=0.4',
    'numpy',
    'pandas >=0.23',
]

version = re.findall(r"""__version__ = ["'](\S+)["']""", open("perfana/__init__.py").read())[0]

setup(
    name=PACKAGE_NAME,
    license='MIT',
    version=version,
    description='Toolbox for performance and portfolio analytics',
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',
    author='Daniel Bok',
    author_email='daniel.bok@outlook.com',
    packages=find_packages(include=['perfana', 'perfana.*']),
    url='https://github.com/DanielBok/perfana',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Financial and Insurance Industry',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7'
    ],
    install_requires=install_requires,
    extras_require={
        "plot": [
            "matplotlib"
        ]
    },
    python_requires='>=3.7',
    zip_safe=False
)
