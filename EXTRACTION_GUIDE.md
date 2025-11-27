# Shamela Books Extraction Guide

This guide explains how to extract the 63 authoritative Islamic texts from Hugging Face datasets.

## 📋 Overview

We're working with two Hugging Face datasets:
- **Metadata**: `MoMonir/Shamela_Books_info` (book titles, authors, IDs, etc.)
- **Full Text**: `MoMonir/shamela_books_text_full` (complete book content)

Our goal is to extract only the 63 books listed in `books.md`.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# OR using uv (recommended)
uv pip install -r requirements.txt
```

### 2. Run the Extraction Script

```bash
# Using uv
uv run extract_books.py

# OR using python directly
python extract_books.py
```

### 3. Check Results

After running, you'll find:
- `output/selected_books.csv` - Metadata for all extracted books
- `output/books/` - Individual markdown files for each book

## 📚 The 63 Books (Golden Subset)

### 1. Tafsir (Quranic Exegesis) - 14 books
- تفسير الطبري
- تفسير القرآن العظيم (Ibn Kathir)
- الجامع لأحكام القرآن (Al-Qurtubi)
- And 11 more...

### 2. Hadith (Prophetic Traditions) - 21 books
- صحيح البخاري
- صحيح مسلم
- سنن أبي داود
- And 18 more...

### 3. Fiqh & Usul (Jurisprudence) - 17 books
- الأم (Al-Shafi'i)
- المغني (Ibn Qudamah)
- بداية المجتهد (Ibn Rushd)
- And 14 more...

### 4. Tarikh & Rijal (History & Biography) - 11 books
- تاريخ الطبري
- البداية والنهاية (Ibn Kathir)
- سير أعلام النبلاء (Al-Dhahabi)
- And 8 more...

## 🔧 How It Works

The `extract_books.py` script performs these steps:

1. **Download** - Fetches both datasets from Hugging Face
2. **Explore** - Analyzes the dataset structure and columns
3. **Filter** - Searches for the 63 books using title matching
4. **Merge** - Combines metadata with full text data
5. **Save** - Exports to CSV and individual markdown files

## 📊 Output Structure

```
output/
├── selected_books.csv          # Master metadata file
└── books/                      # Individual book files
    ├── 01_تفسير_الطبري.md
    ├── 02_صحيح_البخاري.md
    └── ...
```

## 🛠️ Standard Workflow for HF Datasets

When working with Hugging Face datasets, the standard approach is:

1. **Use the `datasets` library** (not manual scraping)
   ```python
   from datasets import load_dataset
   dataset = load_dataset("username/dataset-name")
   ```

2. **Convert to pandas for manipulation**
   ```python
   import pandas as pd
   df = pd.DataFrame(dataset['train'])
   ```

3. **Filter and process** your data
   ```python
   filtered = df[df['column'].isin(your_selection)]
   ```

4. **Save in appropriate format**
   - CSV for metadata/tables
   - Parquet for large datasets (more efficient)
   - JSON for structured data
   - Markdown for readable text

## 🔍 Troubleshooting

### Books Not Found?
- Check if book titles match exactly
- The script uses partial matching, but titles must be similar
- Review the "Not found" messages and check `books.md` for typos

### Memory Issues?
- The datasets are large; ensure you have enough RAM
- Consider downloading in batches
- Use streaming mode: `load_dataset(..., streaming=True)`

### Column Names Different?
- The script tries to auto-detect columns
- If it fails, manually specify column names in the code

## 📝 Next Steps

After extraction:
1. Verify all 63 books were found
2. Check sample books for quality
3. Review metadata in the CSV
4. Begin your analysis/processing pipeline

## 💡 Tips

- **Keep Parquet format** for large data (more efficient than CSV)
- **Use batch processing** if working with many books
- **Cache datasets** to avoid re-downloading
- **Version control** your book selections (keep `books.md` updated)
