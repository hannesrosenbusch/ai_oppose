from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='ai_oppose',
    version='0.0.15',
    author='Hannes Rosenbusch',
    author_email='h.rosenbusch@uva.nl',
    description='generating adversarial takes on science claims with openai plus vectorstores',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hannesrosenbusch/ai_oppose',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    # package_dir={'': 'src'},
    # packages=find_packages(where='src'),
    python_requires='>=3.9',
    install_requires=[
        'google-search-results==2.4.2',
        'googlesearch-python==1.2.3',
        'litstudy==1.0.5',
        'scholarly==1.7.11',
        'selenium==4.11.2',
        'reportlab==4.0.4',
        'fuzzywuzzy==0.18.0',
        'PyPDF2==3.0.1',
        'chromadb==0.4.5',
        'langchain==0.0.257',
        'tiktoken==0.4.0',
        'Unidecode==1.3.6',
        'unstructured==0.9.0',
        'openai==0.27.8',
        'pandas',
    ],
    include_package_data=True,
    package_data={
    'ai_oppose': ['data/*.csv'],
},

)
