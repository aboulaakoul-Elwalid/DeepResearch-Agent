# Shamela Book Extraction - Three Approaches Comparison

## The Problem

You want to extract **63 specific books** from a massive Hugging Face dataset containing **8,492+ books**.

**Issue**: Downloading the entire text dataset causes **OOM (Out of Memory)** errors.

---

## Solution Options

### ✅ Option 1: Smart Streaming Approach (RECOMMENDED)

**File**: `extract_books_smart.py`

**How it works**:
1. Download metadata first (small, ~8MB) ✓ Already done!
2. Identify the 63 book IDs from metadata
3. **Stream** the text dataset (not loading everything into RAM)
4. Save each matching book immediately
5. Stop early once all 63 books are found

**Advantages**:
- ✅ No OOM issues (processes one book at a time)
- ✅ Can resume if interrupted
- ✅ Stops early when all books are found
- ✅ Progress bar shows real-time status
- ✅ Works reliably even on low-memory systems

**Disadvantages**:
- ⏱️ May need to scan many books to find all 63
- 🌐 Requires stable internet (but can resume)

**Usage**:
```bash
python3 extract_books_smart.py
```

**Memory Usage**: ~500MB max (constant, regardless of dataset size)

---

### ⭐ Option 2: Direct Parquet Download (FASTEST)

**Manual approach using HF's parquet files**

**How it works**:
1. Download metadata (small parquet file)
2. Identify the 63 book row indices
3. Download text dataset's parquet file directly
4. Use pandas to filter only the 63 rows

**Advantages**:
- ✅ Very fast (direct file download)
- ✅ No streaming needed
- ✅ Uses efficient parquet format
- ✅ Can filter in pandas easily

**Disadvantages**:
- ⚠️ Still downloads the full text parquet (but it's compressed)
- ⚠️ Need ~2-5GB disk space temporarily
- ⚠️ Requires finding the parquet URL from HF

**Usage**:
```python
import pandas as pd
import requests

# Get parquet URL from HF dataset page
metadata_url = "https://huggingface.co/datasets/MoMonir/Shamela_Books_info/resolve/main/data/train-00000-of-00001.parquet"
text_url = "https://huggingface.co/datasets/MoMonir/shamela_books_text_full/resolve/main/data/train-XXXXX.parquet"

# Download and filter
metadata_df = pd.read_parquet(metadata_url)
text_df = pd.read_parquet(text_url)

# Filter to your 63 books
filtered = text_df[text_df['id'].isin(your_book_ids)]
```

---

### 🔧 Option 3: HF Dataset Viewer API (EXPERIMENTAL)

**File**: `extract_via_api.py` (incomplete)

**How it works**:
1. Use Hugging Face's Dataset Viewer API
2. Query specific rows without downloading
3. Fetch only the 63 books via HTTP requests

**Advantages**:
- ✅ Zero full download
- ✅ Minimal bandwidth usage
- ✅ Fast for small selections

**Disadvantages**:
- ⚠️ API may have rate limits
- ⚠️ Not officially documented for all datasets
- ⚠️ May not work for all dataset configurations
- ⚠️ Requires knowing exact row numbers beforehand

**Status**: Experimental, may not work reliably

---

## 📊 Comparison Table

| Feature | Streaming (Option 1) | Direct Parquet (Option 2) | API (Option 3) |
|---------|---------------------|--------------------------|----------------|
| Memory Usage | ✅ Low (~500MB) | ⚠️ Medium (~2-5GB) | ✅ Low |
| Speed | ⏱️ Medium-Slow | ✅ Fast | ✅ Very Fast |
| Reliability | ✅ High | ✅ High | ⚠️ Medium |
| Network Usage | ⏱️ Continuous | 📥 One big download | ✅ Minimal |
| Resumable | ✅ Yes | ⚠️ Partial | ❌ No |
| Setup Complexity | ✅ Easy | ⚠️ Manual URLs | ⚠️ Complex |

---

## 🎯 Our Recommendation

**Use Option 1: Smart Streaming** (`extract_books_smart.py`)

### Why?
1. **Already set up** - Script is ready to run
2. **Guaranteed to work** - No OOM issues
3. **Progress tracking** - See real-time status
4. **Safe** - Saves books as they're found
5. **Stops early** - Won't scan entire dataset unnecessarily

### Run it now:
```bash
cd /home/elwalid/projects/parallax_project
python3 extract_books_smart.py
```

---

## 🔄 Workflow: Separating Metadata & Text

You asked about separating metadata from text - **we're already doing this!**

### Current Approach (Smart):

```
Step 1: Download Metadata Only
├─ Dataset: MoMonir/Shamela_Books_info
├─ Size: ~8MB (small!)
├─ Contains: Book titles, authors, IDs
└─ Cache: output/metadata/shamela_info.parquet

Step 2: Identify Target Books
├─ Search metadata for your 63 books
├─ Extract book IDs and row indices
└─ Save: output/metadata/matched_books.json

Step 3: Stream Text Dataset
├─ Dataset: MoMonir/shamela_books_text_full
├─ Method: Streaming (not full download)
├─ Match: Check each book's ID against our 63 targets
├─ Action: Save immediately when found
└─ Output: output/books/01_book_title.md

Step 4: Create Master CSV
├─ Combine metadata + extraction status
└─ Save: output/selected_books.csv
```

### Why This Works:
- ✅ **Metadata is small** - Downloads fast, cached locally
- ✅ **Text is streamed** - No OOM, processed incrementally
- ✅ **Separation of concerns** - Identify first, download second
- ✅ **Efficient** - Only downloads what we need (in streaming fashion)

---

## 🚀 Quick Start (Right Now)

```bash
# Make sure you're in the project directory
cd /home/elwalid/projects/parallax_project

# Run the smart streaming extractor
python3 extract_books_smart.py

# It will:
# 1. Load cached metadata (already downloaded)
# 2. Identify your 63 books
# 3. Stream the text dataset
# 4. Save books as they're found
# 5. Show progress bar
# 6. Stop early when all 63 are found
```

---

## 📁 Expected Output

```
parallax_project/
└── output/
    ├── selected_books.csv              # Master metadata file
    ├── metadata/
    │   ├── shamela_info.parquet       # Cached metadata (8,492 books)
    │   └── matched_books.json         # Your 63 book IDs
    └── books/                         # Individual book files
        ├── 01_تفسير_الطبري.md
        ├── 02_صحيح_البخاري.md
        ├── 03_صحيح_مسلم.md
        └── ... (63 files total)
```

---

## 💡 Key Insight: Chunking vs Streaming

You asked about "downloading in chunks" - here's the clarification:

### Chunks (Traditional)
```python
# Downloads in batches
for i in range(0, total, chunk_size):
    chunk = download_data(start=i, end=i+chunk_size)
    process(chunk)
    # Still loads chunks into memory
```

### Streaming (Better)
```python
# Processes one item at a time
for item in stream_data():
    process(item)
    # Only one item in memory at a time
```

**Our approach uses streaming**, which is even better than chunking because memory usage stays constant (one book at a time) regardless of dataset size.

---

## 🎓 Standard Practice Summary

When working with HF datasets where you need a **small subset** from a **large dataset**:

1. ✅ **Download metadata separately** (always small)
2. ✅ **Identify target items** from metadata
3. ✅ **Use streaming** for large text data
4. ✅ **Save incrementally** (don't wait for everything)
5. ✅ **Cache metadata** (avoid re-downloading)
6. ✅ **Use parquet** for storage (efficient)

This is exactly what `extract_books_smart.py` does!

---

## ⚡ Ready to Run?

```bash
python3 extract_books_smart.py
```

Estimated time: 30-60 minutes (depends on network and how early we find all 63 books)

The script will show:
- ✅ Which books are found
- ⏱️ Progress bar with stats
- 💾 Real-time saving (won't lose progress)
- 📊 Final summary
