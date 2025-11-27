# Hugging Face Dataset Workflow - Standard Practice

## Overview

When working with Hugging Face datasets, there's a well-established workflow that data scientists and ML engineers follow. This document outlines the standard practices for your Islamic texts extraction project.

## 📚 Your Project Context

**Goal**: Extract 63 authoritative Islamic texts from Shamela library datasets

**Datasets**:
- `MoMonir/Shamela_Books_info` - Metadata (8,492 books)
- `MoMonir/shamela_books_text_full` - Full text content

**Target**: 63 specific books listed in `books.md`

## 🔄 Standard Workflow

### Phase 1: Setup & Installation

```bash
# Install required packages
pip install datasets pandas pyarrow huggingface-hub

# Or with uv (faster)
uv pip install datasets pandas pyarrow huggingface-hub
```

### Phase 2: Download & Explore

```python
from datasets import load_dataset
import pandas as pd

# Download datasets
info_dataset = load_dataset("MoMonir/Shamela_Books_info")
text_dataset = load_dataset("MoMonir/shamela_books_text_full")

# Convert to pandas for easier manipulation
info_df = pd.DataFrame(info_dataset['train'])
text_df = pd.DataFrame(text_dataset['train'])

# Explore structure
print(info_df.columns)
print(info_df.head())
```

### Phase 3: Filter & Select

```python
# Define your selection criteria
selected_titles = [
    "صحيح البخاري",
    "صحيح مسلم",
    # ... your 63 books
]

# Filter using title matching
filtered_df = info_df[info_df['title'].isin(selected_titles)]

# Or use partial matching
filtered_df = info_df[info_df['title'].str.contains('|'.join(selected_titles))]
```

### Phase 4: Merge & Enrich

```python
# Merge metadata with full text
# Assuming both datasets have a common 'id' or 'book_id' column
merged_df = pd.merge(
    filtered_df,
    text_df,
    on='id',  # or 'book_id', 'bookId', etc.
    how='inner'
)
```

### Phase 5: Export & Save

```python
# Save as CSV (good for metadata)
merged_df.to_csv('output/books_metadata.csv', index=False)

# Save as Parquet (better for large text data)
merged_df.to_parquet('output/books_data.parquet', index=False)

# Save individual books as markdown
for idx, row in merged_df.iterrows():
    with open(f'output/books/{row["title"]}.md', 'w') as f:
        f.write(f"# {row['title']}\n\n")
        f.write(row['text'])
```

## 🎯 Best Practices for HF Datasets

### 1. Use the Datasets Library (Not Web Scraping)

❌ **Don't**: Scrape HTML from Hugging Face website
✅ **Do**: Use `datasets.load_dataset()`

**Why**: It's official, efficient, handles caching, and supports streaming.

### 2. Handle Large Datasets Efficiently

```python
# For very large datasets, use streaming
dataset = load_dataset("username/dataset", streaming=True)

# Or download to disk first
dataset.save_to_disk("./local_cache")
dataset = load_from_disk("./local_cache")
```

### 3. Choose the Right Storage Format

| Format   | Use Case                           | Size  | Speed |
|----------|-----------------------------------|-------|-------|
| CSV      | Metadata, small tables            | Large | Slow  |
| Parquet  | Large datasets, binary data       | Small | Fast  |
| JSON     | Structured/nested data            | Large | Medium|
| Markdown | Human-readable text (like books)  | Large | Medium|

**For your project**: 
- Use **CSV** for the master metadata file
- Use **Markdown** for individual books (readable)
- Use **Parquet** for the full merged dataset (efficient)

### 4. Cache & Version Control

```python
# HF automatically caches downloads in ~/.cache/huggingface/
# You can specify custom cache location
dataset = load_dataset("username/dataset", cache_dir="./my_cache")

# For reproducibility, pin specific versions
dataset = load_dataset("username/dataset", revision="main")
```

### 5. Handle Errors Gracefully

```python
try:
    dataset = load_dataset("username/dataset")
except Exception as e:
    print(f"Error: {e}")
    # Fallback: try loading from local cache
    dataset = load_from_disk("./backup")
```

## 📊 Your Current Progress

✅ Created extraction script (`extract_books.py`)
✅ Downloaded metadata (8,492 books)
⏳ Text dataset download in progress (interrupted at 41%)

## 🚀 Next Steps

### Option A: Resume Download (Recommended)

```bash
# The dataset library should resume from where it stopped
python3 extract_books.py
```

HF automatically caches partial downloads, so it won't start from scratch.

### Option B: Streaming Approach (For Slow Networks)

Modify the script to use streaming:

```python
text_dataset = load_dataset(
    "MoMonir/shamela_books_text_full",
    streaming=True
)

# Process iteratively
for book in text_dataset['train']:
    if book['title'] in selected_titles:
        # Save immediately
        save_book(book)
```

### Option C: Download Metadata First, Text Later

Since you already have metadata (8,492 books):
1. First, identify the exact IDs of your 63 books from metadata
2. Then, selectively download only those 63 texts (if API supports it)
3. This avoids downloading the entire massive text dataset

## 🔍 Troubleshooting

### Network Errors
- **Cause**: Large datasets, unstable connection
- **Solution**: Use streaming mode or increase timeout
- **Alternative**: Download on a stable connection, then transfer files

### Memory Issues
- **Cause**: Loading entire dataset into RAM
- **Solution**: Use `datasets` format (lazy loading) or streaming
- **Code**:
  ```python
  dataset = load_dataset("name", keep_in_memory=False)
  ```

### Missing Columns
- **Cause**: Dataset structure different than expected
- **Solution**: Explore with `dataset.features` first
- **Code**:
  ```python
  print(dataset.features)
  print(dataset['train'][0].keys())
  ```

## 💡 Pro Tips

1. **Always explore before processing**: Run `.head()`, `.info()`, `.describe()` first
2. **Use batch processing**: Don't load everything at once
3. **Keep raw data separate**: Save filtered results in a new directory
4. **Document your filtering logic**: Comment why you selected certain books
5. **Version your outputs**: Name files with dates (`books_2024-01-15.csv`)

## 📁 Recommended Project Structure

```
parallax_project/
├── books.md                    # Your 63-book list (source of truth)
├── extract_books.py            # Main extraction script
├── requirements.txt            # Dependencies
├── cache/                      # HF dataset cache (optional)
├── output/
│   ├── selected_books.csv      # Master metadata
│   ├── selected_books.parquet  # Full data (efficient format)
│   └── books/                  # Individual markdown files
│       ├── 01_sahih_bukhari.md
│       ├── 02_sahih_muslim.md
│       └── ...
└── docs/
    ├── WORKFLOW.md            # This file
    └── EXTRACTION_GUIDE.md    # How-to guide
```

## 🎓 Key Takeaways

1. **Use official APIs** - Don't scrape when there's an API
2. **Datasets library is standard** - It's the go-to for HF datasets
3. **Pandas for manipulation** - Convert to DataFrame for filtering
4. **Parquet for storage** - More efficient than CSV for large data
5. **Cache intelligently** - HF handles this automatically
6. **Stream for huge datasets** - Avoid memory issues
7. **Filter early** - Don't download what you don't need (if possible)

## 📚 Resources

- [Hugging Face Datasets Docs](https://huggingface.co/docs/datasets)
- [Datasets Library Tutorial](https://huggingface.co/docs/datasets/tutorial)
- [Working with Large Datasets](https://huggingface.co/docs/datasets/process)

---

**Current Status**: Ready to resume dataset download or implement streaming approach.
**Recommended Next Action**: Run `python3 extract_books.py` to resume download.
