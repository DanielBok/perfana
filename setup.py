from setuptools import find_packages, setup

import versioneer

PACKAGE_NAME = 'perfana'

cmdclass = versioneer.get_cmdclass()

install_requires = [
    'copulae >= 0.4',
    'numpy',
    'pandas >=0.23',
    'plotly >=3.9'
]

setup(
    name=PACKAGE_NAME,
    license='MIT',
    version=versioneer.get_version(),
    description='Toolbox for performance and portfolio analytics',
    author='Daniel Bok',
    author_email='daniel.bok@outlook.com',
    packages=find_packages(include=['perfana', 'perfana.*']),
    url='https://github.com/DanielBok/perfana',
    cmdclass=cmdclass,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Financial and Insurance Industry',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    install_requires=install_requires,
    python_requires='>=3.6',
    zip_safe=False
)
