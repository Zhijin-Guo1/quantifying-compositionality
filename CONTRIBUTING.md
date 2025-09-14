# Contributing to Quantifying Compositionality

## Setting Up Development Environment

1. Fork and clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download required data (see README.md)

## Data Files Not in Repository

The following large files are NOT included in the git repository and must be downloaded separately:

### Word Experiment
- `data/GoogleNews-vectors-negative300.bin.gz` (~1.5 GB)
- Download from links in README.md

### KG Experiment
- `data/ml-1m/` (MovieLens dataset)
- Download from GroupLens website

These files are listed in `.gitignore` to prevent accidental commits.

## Before Committing

1. Ensure no large data files are staged
2. Run experiments to verify changes work
3. Update documentation if needed

## Code Style

- Use descriptive variable names
- Add docstrings to functions
- Follow existing code patterns
- Keep commits focused and descriptive