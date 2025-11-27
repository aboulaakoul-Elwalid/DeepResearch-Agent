#!/usr/bin/env python3
"""
Smart Shamela Book Extractor - Memory Efficient Version

This script uses a smarter approach:
1. Download metadata first (small, ~8MB)
2. Identify the 63 book IDs from metadata
3. Stream the text dataset and only save matching books
4. No OOM issues - processes one book at a time

Usage:
    python extract_books_smart.py
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# List of 63 books to extract (Arabic titles from books.md)
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


def setup_directories():
    """Create output directories if they don't exist."""
    dirs = ["output", "output/books", "output/metadata"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Output directories ready")


def download_metadata() -> pd.DataFrame:
    """
    Step 1: Download only metadata (small dataset, ~8MB).
    This contains book titles, authors, IDs, etc.
    """
    print("\n" + "=" * 80)
    print("STEP 1: Downloading Metadata (Small)")
    print("=" * 80)

    metadata_cache = Path("output/metadata/shamela_info.parquet")

    if metadata_cache.exists():
        print(f"✓ Loading cached metadata from {metadata_cache}")
        info_df = pd.read_parquet(metadata_cache)
    else:
        print("📥 Downloading shamela_books_info...")
        info_dataset = load_dataset("MoMonir/Shamela_Books_info")
        info_df = pd.DataFrame(info_dataset["train"])

        # Cache it for future runs
        info_df.to_parquet(metadata_cache, index=False)
        print(f"✓ Cached metadata to {metadata_cache}")

    print(f"✓ Loaded {len(info_df)} book metadata entries")
    print(f"✓ Columns: {list(info_df.columns)}")

    return info_df


def find_title_column(df: pd.DataFrame) -> str:
    """Auto-detect which column contains book titles."""
    candidates = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ["title", "name", "book", "عنوان"]):
            candidates.append(col)

    if candidates:
        # Prefer exact matches
        if "title" in candidates:
            return "title"
        return candidates[0]

    # Fallback: use first string column
    for col in df.columns:
        if df[col].dtype == "object":
            return col

    return df.columns[0]


def identify_target_books(info_df: pd.DataFrame, selected_books: List[str]) -> Dict:
    """
    Step 2: Identify which books we need and their IDs.
    Returns a mapping of book IDs to metadata.
    """
    print("\n" + "=" * 80)
    print("STEP 2: Identifying Target Books")
    print("=" * 80)

    title_col = find_title_column(info_df)
    print(f"🔍 Using column '{title_col}' for book titles")

    # Find ID column
    id_col = None
    for col in ["id", "book_id", "bookId", "ID", "Id"]:
        if col in info_df.columns:
            id_col = col
            break

    if not id_col:
        print("⚠️  No ID column found, using index")
        info_df["_index_id"] = info_df.index
        id_col = "_index_id"

    print(f"🔍 Using column '{id_col}' for book IDs")

    # Match books
    matched_books = {}
    not_found = []

    for book_title in selected_books:
        # Try exact match first
        mask = info_df[title_col].astype(str) == book_title
        matches = info_df[mask]

        # If not found, try partial match
        if len(matches) == 0:
            mask = (
                info_df[title_col]
                .astype(str)
                .str.contains(book_title, na=False, regex=False)
            )
            matches = info_df[mask]

        if len(matches) > 0:
            book_data = matches.iloc[0]
            book_id = book_data[id_col]
            matched_books[str(book_id)] = {
                "id": int(book_id),
                "title": str(book_data[title_col]),
                "metadata": {
                    k: int(v)
                    if isinstance(v, (pd.Int64Dtype, type(pd.NA)))
                    or (hasattr(v, "dtype") and "int" in str(v.dtype))
                    else (
                        str(v)
                        if not isinstance(v, (int, float, bool, type(None)))
                        else v
                    )
                    for k, v in book_data.to_dict().items()
                },
            }
            print(f"✓ Found: {book_title} (ID: {book_id})")
        else:
            not_found.append(book_title)
            print(f"✗ Not found: {book_title}")

    print(f"\n📊 Summary:")
    print(f"  Requested: {len(selected_books)} books")
    print(f"  Found: {len(matched_books)} books")
    print(f"  Not found: {len(not_found)} books")

    if not_found:
        print(f"\n⚠️  Missing books:")
        for title in not_found:
            print(f"    - {title}")

    # Save matched book IDs for reference (convert numpy/pandas types to native Python)
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, pd.Int64Dtype)):
            return int(obj)
        elif isinstance(obj, (np.floating, pd.Float64Dtype)):
            return float(obj)
        elif isinstance(obj, (np.bool_, pd.BooleanDtype)):
            return bool(obj)
        elif pd.isna(obj):
            return None
        elif hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        return obj

    serializable_books = convert_to_native(matched_books)
    with open("output/metadata/matched_books.json", "w", encoding="utf-8") as f:
        json.dump(serializable_books, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved matched books to output/metadata/matched_books.json")

    return matched_books


def stream_and_extract_books(matched_books: Dict):
    """
    Step 3: Stream the text dataset and extract only the 63 books we need.
    This prevents OOM by processing one book at a time.
    """
    print("\n" + "=" * 80)
    print("STEP 3: Streaming Text Dataset (Memory Efficient)")
    print("=" * 80)
    print("⚠️  This may take a while, but won't cause OOM issues")

    target_ids = set(matched_books.keys())
    found_books = {}

    print(f"\n📥 Streaming dataset: MoMonir/shamela_books_text_full")
    print(f"🎯 Looking for {len(target_ids)} books...")

    try:
        # Stream the dataset (doesn't load everything into memory)
        text_dataset = load_dataset(
            "MoMonir/shamela_books_text_full", streaming=True, split="train"
        )

        # Process iteratively
        processed = 0
        with tqdm(desc="Processing books", unit=" books") as pbar:
            for book in text_dataset:
                processed += 1
                pbar.update(1)

                # Check if this is one of our target books
                book_id = None
                for id_field in ["id", "book_id", "bookId", "ID"]:
                    if id_field in book:
                        book_id = str(book[id_field])
                        break

                if book_id and book_id in target_ids:
                    found_books[book_id] = book
                    target_ids.remove(book_id)

                    # Save immediately (one book at a time)
                    metadata = matched_books[book_id]
                    save_single_book(book, metadata, len(found_books))

                    tqdm.write(
                        f"✓ Saved: {metadata['title']} ({len(found_books)}/{len(matched_books)})"
                    )

                    # Stop early if we found all books
                    if len(target_ids) == 0:
                        tqdm.write("\n🎉 All books found! Stopping early.")
                        break

                # Progress update every 100 books
                if processed % 100 == 0:
                    pbar.set_postfix(
                        {
                            "found": len(found_books),
                            "remaining": len(target_ids),
                            "processed": processed,
                        }
                    )

    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")
        print(f"✓ Saved {len(found_books)} books before error")

    print(f"\n📊 Final Summary:")
    print(f"  Requested: {len(matched_books)} books")
    print(f"  Found: {len(found_books)} books")
    print(f"  Processed: {processed} total books")

    if target_ids:
        print(f"\n⚠️  Still missing {len(target_ids)} books:")
        for book_id in list(target_ids)[:10]:  # Show first 10
            print(f"    - {matched_books[book_id]['title']}")

    return found_books


def save_single_book(book_data: Dict, metadata: Dict, index: int):
    """Save a single book to markdown file."""
    title = metadata["title"]

    # Clean filename
    safe_title = "".join(
        c for c in title if c.isalnum() or c in (" ", "-", "_")
    ).strip()
    safe_title = safe_title.replace(" ", "_")[:100]

    book_path = Path(f"output/books/{index:02d}_{safe_title}.md")

    # Find text column
    text = None
    for col in ["text", "content", "book_text", "full_text", "نص"]:
        if col in book_data:
            text = book_data[col]
            break

    if text is None:
        # Fallback: use longest string value
        text = max(
            (v for v in book_data.values() if isinstance(v, str)), key=len, default=""
        )

    # Write markdown file
    with open(book_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"**Book ID:** {metadata['id']}\n")
        f.write(f"**Index:** {index}\n\n")
        f.write("---\n\n")
        f.write(str(text))


def create_master_csv(matched_books: Dict, found_books: Dict):
    """Create a master CSV with all metadata."""
    print("\n" + "=" * 80)
    print("STEP 4: Creating Master CSV")
    print("=" * 80)

    records = []
    for book_id, book_info in matched_books.items():
        if book_id in found_books:
            record = book_info["metadata"].copy()
            record["_extracted"] = True
            records.append(record)

    df = pd.DataFrame(records)
    csv_path = "output/selected_books.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"✓ Saved master CSV: {csv_path} ({len(df)} books)")


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("SMART SHAMELA BOOK EXTRACTOR")
    print("Memory-Efficient Streaming Approach")
    print("=" * 80)

    try:
        # Setup
        setup_directories()

        # Step 1: Download metadata (small, fast)
        info_df = download_metadata()

        # Step 2: Identify the 63 target books
        matched_books = identify_target_books(info_df, SELECTED_BOOKS)

        if not matched_books:
            print("\n❌ No books found! Check your book titles.")
            return

        # Step 3: Stream text dataset and extract only target books
        found_books = stream_and_extract_books(matched_books)

        # Step 4: Create master CSV
        if found_books:
            create_master_csv(matched_books, found_books)

        print("\n" + "=" * 80)
        print("✅ EXTRACTION COMPLETE!")
        print("=" * 80)
        print(f"\n📁 Results:")
        print(f"  - Metadata CSV: output/selected_books.csv")
        print(f"  - Book files: output/books/")
        print(f"  - Matched IDs: output/metadata/matched_books.json")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("✓ Partial results saved in output/ directory")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
