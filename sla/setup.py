from setuptools import setup, find_packages

setup(
    name='sla',
    version='0.1',
    packages=find_packages(),
    install_requires=[],
    author='Arq Innovacion',
    author_email='jomaver@bancolombia.com.co',
    description='Analisis de logs en python',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Estrategia-e-innovacion-de-TI/SLA-Analisis-logs-python',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Apache License 2.0',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)
