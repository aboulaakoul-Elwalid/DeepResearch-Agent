#!/usr/bin/env python3
"""
API-Based Shamela Book Extractor - Zero Download Approach

This script uses Hugging Face's Dataset Viewer API to query specific rows
without downloading the entire dataset. Perfect for extracting only 63 books
from a massive dataset.

Benefits:
- No OOM issues
- No large downloads
- Fast and efficient
- Uses HF's parquet files directly

Usage:
    python extract_via_api.py
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

# List of 63 books to extract
SELECTED_BOOKS = [
    # 1. Tafsir (Quranic Exegesis) - 14 books
    "تفسير الطبري",
    "جامع البيان عن تأويل آي القرآن",
    "تفسير القرآن العظيم",
    "الجامع لأحكام القرآن",
    "مفاتيح الغيب",
    "التفسير الكبير",
    "الكشاف عن حقائق غوامض التنزيل وعيون الأقاويل في وجوه التأويل",
    "أنوار التنزيل وأسرار التأويل",
    "معالم التنزيل في تفسير القرآن",
    "تفسير البغوي",
    "روح المعاني في تفسير القرآن العظيم والسبع المثاني",
    "تفسير الجلالين",
    "أضواء البيان في إيضاح القرآن بالقرآن",
    "المحرر الوجيز في تفسير الكتاب العزيز",
    "الدر المنثور",
    "أحكام القرآن",
    "زاد المسير في علم التفسير",
    # 2. Hadith (Prophetic Traditions) - 21 books
    "صحيح البخاري",
    "صحيح مسلم",
    "سنن أبي داود",
    "سنن الترمذي",
    "سنن النسائي المجتبى",
    "سنن ابن ماجه",
    "الموطأ",
    "المستدرك على الصحيحين",
    "صحيح ابن خزيمة",
    "صحيح ابن حبان بترتيب ابن بلبان",
    "المنهاج شرح صحيح مسلم بن الحجاج",
    "فتح الباري",
    "التعليق على فتح الباري",
    "تحفة الأحوذي بشرح جامع الترمذي",
    "عون المعبود",
    "سنن أبي داود مع شرحه عون المعبود",
    "نيل الأوطار",
    "سبل السلام الموصلة إلى بلوغ المرام",
    "رياض الصالحين",
    "الأربعون النووية",
    "عمدة الأحكام الكبرى",
    "مشكاة المصابيح",
    "الترغيب والترهيب",
    # 3. Fiqh & Usul (Jurisprudence & Principles) - 17 books
    "الأم",
    "الرسالة",
    "المدونة",
    "المبسوط",
    "بدائع الصنائع في ترتيب الشرائع",
    "المغني",
    "المجموع شرح المهذب",
    "روضة الطالبين وعمدة المفتين",
    "بداية المجتهد ونهاية المقتصد",
    "كشاف القناع عن الإقناع",
    "الذخيرة",
    "الموافقات",
    "إعلام الموقعين عن رب العالمين",
    "أصول الفقه",
    "البرهان في أصول الفقه",
    "المستصفى",
    "الطرق الحكمية",
    # 4. Tarikh & Rijal (History & Biography) - 11 books
    "تاريخ الطبري",
    "تاريخ الرسل والملوك",
    "الكامل في التاريخ",
    "البداية والنهاية",
    "سير أعلام النبلاء",
    "تاريخ بغداد",
    "الطبقات الكبرى",
    "الإصابة في تمييز الصحابة",
    "أسد الغابة في معرفة الصحابة",
    "تهذيب التهذيب",
    "وفيات الأعيان وأنباء أبناء الزمان",
    "الروض الأنف في شرح السيرة النبوية",
]

# HF API endpoints
HF_API_BASE = "https://datasets-server.huggingface.co"
HF_DATASETS_API = "https://huggingface.co/api/datasets"


def setup_directories():
    """Create output directories."""
    dirs = ["output", "output/books", "output/metadata"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def fetch_metadata_via_api(
    dataset_name: str = "MoMonir/Shamela_Books_info",
) -> "pd.DataFrame":
    """
    Fetch metadata using HF's parquet files API.
    This is much faster than downloading the entire dataset.
    """
    print("\n" + "=" * 80)
    print("STEP 1: Fetching Metadata via API")
    print("=" * 80)

    cache_file = Path("output/metadata/shamela_info.parquet")

    if cache_file.exists():
        print(f"✓ Loading cached metadata from {cache_file}")
        return pd.read_parquet(cache_file)

    print(f"📥 Fetching metadata from {dataset_name}...")

    try:
        # Get parquet files info
        url = f"{HF_DATASETS_API}/{dataset_name}/parquet"
        response = requests.get(url)
        response.raise_for_status()

        parquet_info = response.json()

        # Download the parquet file directly
        if "train" in parquet_info:
            parquet_url = parquet_info["train"][0]["url"]
            print(f"📥 Downloading parquet: {parquet_url}")

            df = pd.read_parquet(parquet_url)
            df.to_parquet(cache_file, index=False)
            print(f"✓ Cached to {cache_file}")
            print(f"✓ Loaded {len(df)} metadata records")

            return df

    except Exception as e:
        print(f"⚠️  API method failed: {e}")
        print("📥 Falling back to datasets library...")

    # Fallback to datasets library
    from datasets import load_dataset

    info_dataset = load_dataset(dataset_name)
    df = pd.DataFrame(info_dataset["train"])
    df.to_parquet(cache_file, index=False)
    print(f"✓ Loaded {len(df)} metadata records")

    return df


def query_rows_api(
    dataset_name: str, offset: int = 0, length: int = 100
) -> Optional[Dict]:
    """
    Query specific rows from HF dataset using the viewer API.
    """
    url = f"{HF_API_BASE}/rows"
    params = {
        "dataset": dataset_name,
        "config": "default",
        "split": "train",
        "offset": offset,
        "length": length,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"⚠️  API query failed at offset {offset}: {e}")
        return None


def search_api(dataset_name: str, query: str) -> Optional[Dict]:
    """
    Search dataset using HF's search API (if available).
    """
    url = f"{HF_API_BASE}/search"
    params = {
        "dataset": dataset_name,
        "config": "default",
        "split": "train",
        "query": query,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except:
        return None


def identify_books_from_metadata(
    metadata_df: pd.DataFrame, selected_books: List[str]
) -> Dict:
    """
    Identify target books from metadata.
    Returns mapping of indices to book info.
    """
    print("\n" + "=" * 80)
    print("STEP 2: Identifying Target Books")
    print("=" * 80)

    # Find title column
    title_col = None
    for col in metadata_df.columns:
        if any(kw in col.lower() for kw in ["title", "name", "book", "عنوان"]):
            title_col = col
            break

    if not title_col:
        title_col = metadata_df.columns[0]

    print(f"🔍 Using column '{title_col}' for titles")

    matched_books = {}
    not_found = []

    for book_title in selected_books:
        # Try exact match
        mask = metadata_df[title_col].astype(str) == book_title
        matches = metadata_df[mask]

        # Try partial match
        if len(matches) == 0:
            mask = (
                metadata_df[title_col]
                .astype(str)
                .str.contains(book_title, na=False, regex=False)
            )
            matches = metadata_df[mask]

        if len(matches) > 0:
            idx = matches.index[0]
            row = matches.iloc[0]

            matched_books[idx] = {
                "index": idx,
                "title": row[title_col],
                "metadata": row.to_dict(),
            }
            print(f"✓ Found: {book_title} (row index: {idx})")
        else:
            not_found.append(book_title)
            print(f"✗ Not found: {book_title}")

    print(f"\n📊 Summary:")
    print(f"  Requested: {len(selected_books)}")
    print(f"  Found: {len(matched_books)}")
    print(f"  Missing: {len(not_found)}")

    if not_found and len(not_found) <= 10:
        print(f"\n⚠️  Missing books:")
        for title in not_found:
            print(f"    - {title}")

    # Save matched info
    with open("output/metadata/matched_books.json", "w", encoding="utf-8") as f:
        json.dump(matched_books, f, ensure_ascii=False, indent=2, default=str)

    return matched_books


def fetch_book_by_index(
    dataset_name: str,
    row_index: int,
    batch_size: int = 100,
) -> Optional[Dict]:
    """
    Fetch a specific book by its row index using the API.
    Uses batch queries centered around the target index.
    """
    offset = max(0, row_index - (batch_size // 2))
    result = query_rows_api(dataset_name, offset=offset, length=batch_size)

    if not result or "rows" not in result:
        return None

    for row in result["rows"]:
        if row.get("row_idx") == row_index:
            return row.get("row", {})

    # Try direct fetch if not in batch
    direct = query_rows_api(dataset_name, offset=row_index, length=1)
    if direct and "rows" in direct and len(direct["rows"]) > 0:
        return direct["rows"][0].get("row", {})

    return None


def save_single_book(book_data: Dict, book_info: Dict, output_dir: Path) -> bool:
    """
    Save a single book's content and metadata.
    """
    book_id = book_info.get("index", "unknown")
    title = book_info.get("title", "untitled")

    # Create safe filename
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in str(title))
    safe_title = safe_title[:50]  # Limit length

    book_dir = output_dir / str(book_id)
    book_dir.mkdir(parents=True, exist_ok=True)

    # Save content
    content = book_data.get("text", "") or book_data.get("content", "")
    if not content:
        # Try to find any text field
        for key, value in book_data.items():
            if isinstance(value, str) and len(value) > 100:
                content = value
                break

    if content:
        content_file = book_dir / f"{book_id}_{safe_title}.txt"
        content_file.write_text(content, encoding="utf-8")

    # Save metadata
    meta = {
        "book_id": book_id,
        "title": title,
        "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **book_info.get("metadata", {}),
    }
    meta_file = book_dir / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return bool(content)


def extract_books_via_api(
    dataset_name: str = "MoMonir/shamela_books",
    matched_books: Optional[Dict] = None,
    output_dir: Path = Path("output/books"),
) -> Dict[str, bool]:
    """
    Extract matched books using the HuggingFace API.
    """
    print("\n" + "=" * 80)
    print("STEP 3: Extracting Books via API")
    print("=" * 80)

    if matched_books is None:
        matched_file = Path("output/metadata/matched_books.json")
        if matched_file.exists():
            with open(matched_file, "r", encoding="utf-8") as f:
                matched_books = json.load(f)
        else:
            print("No matched books found. Run identification first.")
            return {}

    if not matched_books:
        print("No matched books to extract.")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, bool] = {}

    for idx, info in tqdm(matched_books.items(), desc="Extracting books"):
        row_index = int(idx)
        title = info.get("title", "unknown")

        # Check if already extracted
        book_dir = output_dir / str(row_index)
        if book_dir.exists() and any(book_dir.glob("*.txt")):
            print(f"  Skipping {title} (already extracted)")
            results[str(row_index)] = True
            continue

        # Fetch the book
        book_data = fetch_book_by_index(dataset_name, row_index)

        if book_data:
            success = save_single_book(book_data, info, output_dir)
            results[str(row_index)] = success
            if success:
                print(f"  Extracted: {title}")
            else:
                print(f"  No content: {title}")
        else:
            results[str(row_index)] = False
            print(f"  Failed: {title}")

        # Rate limiting
        time.sleep(0.5)

    # Summary
    successful = sum(1 for v in results.values() if v)
    print(f"\n Extraction complete: {successful}/{len(results)} books")

    return results


def create_master_csv(
    matched_books: Dict,
    results: Dict[str, bool],
    output_path: Path = Path("output/metadata/extracted_books.csv"),
) -> None:
    """
    Create a master CSV of extracted books for downstream processing.
    """
    rows = []
    for idx, info in matched_books.items():
        row = {
            "book_id": idx,
            "title": info.get("title", ""),
            "extracted": results.get(str(idx), False),
            "file_path": f"output/books/{idx}/" if results.get(str(idx)) else "",
        }
        # Add any metadata fields
        if "metadata" in info:
            for key, value in info["metadata"].items():
                if key not in row:
                    row[key] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n Master CSV saved to {output_path}")


def main():
    """
    Main entry point for the API-based extraction.
    """
    print("\n" + "=" * 80)
    print("SHAMELA BOOK EXTRACTOR - API MODE")
    print("=" * 80)

    # Setup
    setup_directories()

    # Step 1: Fetch metadata
    metadata_df = fetch_metadata_via_api()
    print(f"\n Metadata loaded: {len(metadata_df)} records")
    print(f"  Columns: {list(metadata_df.columns)}")

    # Step 2: Identify target books
    matched_books = identify_books_from_metadata(metadata_df, SELECTED_BOOKS)

    if not matched_books:
        print("\n No matching books found. Exiting.")
        return

    # Step 3: Extract books via API
    results = extract_books_via_api(matched_books=matched_books)

    # Step 4: Create master CSV
    create_master_csv(matched_books, results)

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"  Books requested: {len(SELECTED_BOOKS)}")
    print(f"  Books matched: {len(matched_books)}")
    print(f"  Books extracted: {sum(1 for v in results.values() if v)}")
    print("\nOutput directories:")
    print("  - output/books/ - Extracted book content")
    print("  - output/metadata/ - Metadata and CSV files")


if __name__ == "__main__":
    main()
