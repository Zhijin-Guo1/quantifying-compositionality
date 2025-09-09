# Knowledge Graph Compositionality Experiments

## Quick Start for KG Experiments

### Prerequisites

1. **MovieLens 1M Dataset**
   - Download from: https://grouplens.org/datasets/movielens/1m/
   - Extract to `ml-1m/` directory
   - Should contain: `users.dat`, `movies.dat`, `ratings.dat`, `README`

2. **Pre-trained KG embeddings** in `KG_embedding/`:
   - `300_epochs_TransE_gpu34.pt` (6040 user embeddings)
   - `300_epochs_DistMult_gpu34.pt` (6040 user embeddings)
   - These embeddings should be trained on MovieLens user-item interactions

### Basic Commands

```bash
# Run with TransE embeddings and MovieLens data
python run_experiments.py --experiment kg --movielens-dir ml-1m --kg-model TransE --plot

# Run with DistMult embeddings
python run_experiments.py --experiment kg --movielens-dir ml-1m --kg-model DistMult --plot

# With normalization
python run_experiments.py --experiment kg --movielens-dir ml-1m --kg-model TransE --normalize-kg --plot
```

### Data Structure

MovieLens 1M contains 6040 users with demographics:
- **Gender**: M (Male), F (Female)
- **Age groups**: 1, 18, 25, 35, 45, 50, 56
- **Occupations**: 21 categories
  - 0: other/not specified
  - 1: academic/educator
  - 2: artist
  - 3: clerical/admin
  - 4: college/grad student
  - 5: customer service
  - 6: doctor/health care
  - 7: executive/managerial
  - 8: farmer
  - 9: homemaker
  - 10: K-12 student
  - 11: lawyer
  - 12: programmer
  - 13: retired
  - 14: sales/marketing
  - 15: scientist
  - 16: self-employed
  - 17: technician/engineer
  - 18: tradesman/craftsman
  - 19: unemployed
  - 20: writer

### How It Works

1. **Automatic loading**: Reads all 6040 users from `users.dat`
2. **Feature encoding**: Creates one-hot encoding (30 features total)
3. **ID mapping**: User 1 → embedding index 0, User 2 → index 1, etc.
4. **Grouping**: Users with identical demographics are grouped for analysis

### Advanced Options

```bash
# Full command with all options
python run_experiments.py \
    --experiment kg \
    --movielens-dir ml-1m \
    --kg-model TransE \
    --kg-embedding-dir KG_embedding \
    --normalize-kg \
    --cca-components 30 \
    --n-permutations 100 \
    --n-trials 100 \
    --plot \
    --save-results \
    --output-dir results/ \
    --verbose
```

### Troubleshooting

1. **"MovieLens data not found"**: Download from https://grouplens.org/datasets/movielens/1m/
2. **"KG embeddings not found"**: Ensure .pt files are in `KG_embedding/` directory
3. **"Dimension mismatch"**: Embeddings must have exactly 6040 users
4. **Memory issues**: Use `--normalize-kg` or reduce `--n-permutations`

### Expected Output

```
KNOWLEDGE GRAPH COMPOSITIONALITY EXPERIMENT
============================================================
Loading MovieLens data from ml-1m...
Loaded 6040 users from MovieLens
Demographics: 30 features
   Attributes shape: (6040, 30)
   Features (first 10): ['gender_F', 'gender_M', 'age_1', 'age_18', 'age_25', ...]

2. Loading TransE embeddings...
   Embeddings shape: (6040, 50)

3. Analyzing compositionality...
   CCA Score: 0.5974
   Decomposition Score: 0.4334
   Overall Compositionality Score: 0.5154

Results saved to output/kg_TransE_results.npz
Plot saved to output/kg_TransE_compositionality.png
```

### Interpreting Results

The compositionality score indicates the alignment between KG embeddings and demographic attributes. Higher scores indicate stronger compositional structure.

### Notes on Results

Based on the paper's findings:
- CCA correlations increase ~1.5× from early to late training stages
- Demographics are partially encoded in behavior-based embeddings
- Grouping users by identical demographics improves signal clarity