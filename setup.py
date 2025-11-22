"""Setup script for AMCF package"""

from setuptools import setup, find_packages

setup(
    name='amcf-framework',
    version='0.1.0',
    description='Adaptive Metacognitive Framework for Proactive Cyber Security',
    author='Evren Daglarli',
    author_email='daglarli@itu.edu.tr',
    url='https://github.com/evrendaglarli/amcf-framework',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'lightgbm>=3.3.0',
        'torch>=1.10.0',
        'sentence-transformers>=2.2.0',
        'hnswlib>=0.7.0',
        'pyyaml>=5.4.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Security',
    ],
)
```

### `requirements.txt`
```
numpy==1.24.3
pandas==2.0.3
scipy==1.11.0
scikit-learn==1.3.0
lightgbm==4.0.0
torch==2.0.1
sentence-transformers==2.2.2
hnswlib==0.7.0
pyyaml==6.0
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0
pytest==7.4.0
pytest-cov==4.1.0
```

### `.gitignore`
```
# Byte-compiled
__pycache__/
*.py[cod]
*$py.class

# Distribution
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Data
data/raw/
data/processed/*.pkl
!data/processed/README.md

# Results
results/tables/*.csv
results/figures/*.png
results/*.log

# Virtual env
venv/
ENV/
env/

# OS
.DS_Store
Thumbs.db