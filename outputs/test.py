# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=amcf --cov-report=html
```

## Documentation

- `docs/architecture.md`: System architecture and design decisions
- `docs/algorithms.md`: Detailed algorithm descriptions and equations
- `docs/reproducibility.md`: Complete reproducibility guide
- `docs/api_reference.md`: API documentation

## Known Limitations

1. **Scalability**: Episodic memory scales poorly beyond 10^7 incidents without dimensionality reduction
2. **Concept drift**: Recalibration lags 3-5 days under rapid drift (>20% feature shift)
3. **Label quality**: Analyst feedback inter-rater agreement κ=0.6-0.7 introduces label noise
4. **Computation**: 2.5× overhead vs. static SIEM correlation

See Section 7 of the paper for full discussion.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contact

**Evren Daglarli**
- Email: daglarli@itu.edu.tr
- Affiliation: Istanbul Technical University, Faculty of Computer and Informatics Engineering
- AI Research Center: ITUAI

## Acknowledgments

- Istanbul Technical University Cognitive Systems Laboratory (CSL)
- MITRE ATT&CK framework for TTP taxonomy
- Contributors to uncertainty quantification and calibration literature

---

**Last Updated**: December 2024
**Paper Status**: Submitted to Big Data and Cognitive Computing
**Repository Status**: Active development
```

### `requirements.txt`
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
torch>=1.10.0
sentence-transformers>=2.2.0
hnswlib>=0.7.0
pyyaml>=5.4.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
pytest>=7.0.0
pytest-cov>=3.0.0
```

### `setup.py`
```python
from setuptools import setup, find_packages

setup(
    name='amcf-framework',
    version='0.1.0',
    description='Adaptive Metacognitive Framework for Proactive Cyber Security',
    author='Evren Daglarli',
    author_email='daglarli@itu.edu.tr',
    url='https://github.com/your-org/amcf-framework',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        line.strip() for line in open('requirements.txt').readlines()
        if line.strip() and not line.startswith('#')
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

---

## 7. RUNNING THE REPOSITORY

### Step 1: Clone and Setup
```bash
git clone https://github.com/your-org/amcf-framework.git
cd amcf-framework
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Step 2: Generate Synthetic Data
```bash
python data/synthetic/generate_synthetic_data.py
```

### Step 3: Run Experiments
```bash
python experiments/run_experiments.py
```

### Step 4: Verify Results
```bash
# Check results match paper
python scripts/verify_results.py
```

### Expected Output