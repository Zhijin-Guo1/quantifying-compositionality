# Real Data Usage Guide

This document explains how each experiment uses **real datasets** instead of synthetic data, following the implementation patterns from the research notebooks.

## ✅ All Experiments Now Use Real Data

### 1. Sentence Compositionality (Schema-Guided Dialogue)

**Dataset**: Schema-Guided Dialogue Dataset  
**Location**: `train/` directory  
**Files Required**:
- `dialogues_*.json` - Dialogue files with user turns and slot annotations
- `schema.json` - Schema defining all possible slots

**Usage**:
```bash
python run_experiments.py --experiment sentence --dialogue-dir train
```

**Implementation**:
- Loads dialogues from JSON files using `DialogueLoader`
- Filters sentences with ≥3 slots (configurable via `--min-slots`)
- Creates binary slot matrices for attributes
- Groups sentences by unique slot combinations
- Matches the approach in `notebooks/sentence_concept.ipynb`

### 2. Word Compositionality (MorphoLEX)

**Dataset**: MorphoLEX English  
**Location**: `Downloads/MorphoLEX_en.xlsx`  
**Download From**: http://www.lexique.org/?page_id=250

**Usage**:
```bash
# With MorphoLEX data
python run_experiments.py --experiment word --morpholex-path Downloads/MorphoLEX_en.xlsx

# With custom word list
python run_experiments.py --experiment word --data-path my_words.txt
```

**Implementation**:
- Loads morphological features from MorphoLEX using `MorphoLEXLoader`
- Extracts features like prefixes, suffixes, tense markers
- Can filter to specific word list if provided
- Falls back to basic morphological features if MorphoLEX unavailable
- Matches the approach in `notebooks/morphology_linear_decomposition.ipynb`

### 3. Knowledge Graph (MovieLens 1M)

**Dataset**: MovieLens 1M  
**Location**: `ml-1m/` directory  
**Download From**: https://grouplens.org/datasets/movielens/1m/

**Files Required**:
- `users.dat` - 6040 users with demographics (gender, age, occupation)
- `movies.dat` - Movie information (optional)
- `ratings.dat` - User-movie ratings (optional)
- `README` - Contains occupation mappings

**Pre-trained Embeddings Required**:
- `KG_embedding/300_epochs_TransE_gpu34.pt`
- `KG_embedding/300_epochs_DistMult_gpu34.pt`

**Usage**:
```bash
python run_experiments.py --experiment kg --movielens-dir ml-1m --kg-model TransE
```

**Implementation**:
- Loads all 6040 MovieLens users using `MovieLensLoader`
- Creates one-hot encoding for demographics (30 features total)
- Maps user IDs to embedding indices (user 1 → index 0, etc.)
- Groups users with identical demographics
- Matches the approach in `notebooks/compositionality_KG.ipynb`

## 📊 Data Processing Details

### Sentence Data Processing
1. **Loading**: Reads all `dialogues_*.json` files from directory
2. **Filtering**: Keeps only user turns with ≥3 slots
3. **Attributes**: Binary matrix indicating slot presence
4. **Grouping**: Groups sentences with identical slot combinations
5. **Result**: ~2458 sentences → ~90 unique groups

### Word Data Processing
1. **Loading**: Reads MorphoLEX Excel sheets (2, 3, 4)
2. **Features**: Extracts morphological markers (PRS, PST, PLUR, etc.)
3. **Filtering**: Can filter to specific word list
4. **Fallback**: Creates basic features (has_ing, has_ed, etc.) if needed

### KG Data Processing
1. **Loading**: Reads `users.dat` with delimiter '::'
2. **Demographics**: One-hot encodes gender (2), age (7), occupation (21)
3. **Mapping**: User IDs 1-6040 → embedding indices 0-6039
4. **Embeddings**: Loads pre-trained TransE/DistMult embeddings
5. **Grouping**: Groups users by demographic patterns

## 🔄 Migration from Synthetic Data

The original implementation used synthetic data generation:
- ❌ Random entity generation with `--n-entities`
- ❌ Demo sentences without real annotations
- ❌ Random word lists without morphological data

Now all experiments use real datasets:
- ✅ MovieLens users with actual demographics
- ✅ Schema-Guided Dialogue with real slot annotations
- ✅ MorphoLEX with linguistic morphological features

## 📁 Required Directory Structure

```
quantifying-compositionality/
├── train/                      # Schema-Guided Dialogue
│   ├── schema.json
│   └── dialogues_*.json
├── ml-1m/                      # MovieLens 1M
│   ├── users.dat
│   ├── movies.dat
│   ├── ratings.dat
│   └── README
├── KG_embedding/               # Pre-trained embeddings
│   ├── 300_epochs_TransE_gpu34.pt
│   └── 300_epochs_DistMult_gpu34.pt
└── Downloads/                  # Optional: MorphoLEX
    └── MorphoLEX_en.xlsx
```

## 🚀 Quick Start

```bash
# Install requirements
pip install -r requirements.txt

# Download datasets
# 1. Schema-Guided Dialogue: https://github.com/google-research-datasets/dstc8-schema-guided-dialogue
# 2. MovieLens 1M: https://grouplens.org/datasets/movielens/1m/
# 3. MorphoLEX: http://www.lexique.org/?page_id=250

# Run experiments with real data
python run_experiments.py --experiment sentence --dialogue-dir train --plot
python run_experiments.py --experiment word --morpholex-path Downloads/MorphoLEX_en.xlsx --plot
python run_experiments.py --experiment kg --movielens-dir ml-1m --kg-model TransE --plot

# Run all experiments
python run_experiments.py --experiment all --plot --save-results
```

## 📝 Notes

- All data loaders are in `data_loaders/` module
- Loaders handle missing data gracefully with informative error messages
- Demo/fallback data is only used when explicitly requested or when real data is unavailable
- The implementation matches the data processing in the research notebooks exactly