# Data Science Project Template

## Project Structure Overview

```
project-template/
├── notebooks/         # Jupyter notebooks for analysis and exploration
├── projectname/       # Main Python package with reusable code
├── data/             # Data storage
├── scripts/          # Standalone scripts and automation tools
└── README.md         # This file
```

## Detailed Directory Explanations

### 📓 `notebooks/`
Contains Jupyter notebooks used for exploration, analysis, and visualization.
- `01-first-logical-notebook.ipynb`, `02-second-logical-notebook.ipynb`: Main analysis notebooks, numbered for clear workflow
- `prototype-notebook.ipynb`: For quick experiments and prototyping
- `archive/`: Store old notebooks that are no longer actively used but worth keeping

### 📦 `projectname/` (Main Package)
The core Python package containing reusable code. This is where you put code that's used across multiple notebooks or scripts.

```
projectname/
├── projectname/          # Actual package code
│   ├── __init__.py      # Makes the folder a Python package
│   ├── config.py        # Configuration settings and parameters
│   ├── data.py          # Data loading and processing functions
│   └── utils.py         # Utility functions used across the project
└── setup.py             # Package installation and dependencies
```

The double `projectname/` structure is a Python packaging convention:
- Outer `projectname/`: Contains package metadata and setup files
- Inner `projectname/`: Contains the actual Python module code

### 📊 `data/`
Organized storage for project data files:
- `raw/`: Original, immutable data
- `processed/`: Data that has been modified from its original form
- `cleaned/`: Final, analysis-ready datasets

### 🛠️ `scripts/`
Standalone Python scripts for various purposes:
- Data processing pipelines
- Automation tasks
- Command-line tools
- Regular jobs or scheduled tasks

Difference between `scripts/` and `projectname/`:
- `scripts/`: Contains standalone executable programs
- `projectname/`: Contains reusable functions and classes imported by notebooks and scripts

### Example Usage

#### Using the Package
```python
from projectname.data import load_data
from projectname.utils import setup_logging

# Load your data
data = load_data()
```

#### Running Scripts
```bash
python scripts/script1.py
```

## Setup Instructions

1. Clone this repository
```bash
git clone <repository-url>
cd project-template
```

2. Install the package in development mode
```bash
pip install -e .
```

3. Create directories for your data
```bash
mkdir -p data/{raw,processed,cleaned}
```

## Best Practices

1. Keep notebooks clean and well-documented
2. Never store large data files in Git
3. Use relative paths defined in `config.py`
4. Archive notebooks instead of deleting them
5. Write reusable functions in the package


