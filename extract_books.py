#!/usr/bin/env python3
"""
Script to download Hugging Face datasets and extract the 63 selected Islamic texts.

This script:
1. Downloads metadata and text datasets from Hugging Face
2. Filters for the 63 books specified in books.md
3. Saves the filtered data to CSV and individual markdown files
"""

import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset

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


def download_datasets():
    """Download both metadata and text datasets from Hugging Face."""
    print("=" * 80)
    print("STEP 1: Downloading datasets from Hugging Face")
    print("=" * 80)

    print("\n📥 Downloading shamela_books_info (metadata)...")
    info_dataset = load_dataset("MoMonir/Shamela_Books_info")
    info_df = pd.DataFrame(info_dataset["train"])
    print(f"✓ Downloaded {len(info_df)} book metadata entries")

    print("\n📥 Downloading shamela_books_text_full (full text)...")
    text_dataset = load_dataset("MoMonir/shamela_books_text_full")
    text_df = pd.DataFrame(text_dataset["train"])
    print(f"✓ Downloaded {len(text_df)} book text entries")

    return info_df, text_df


def explore_structure(info_df, text_df):
    """Explore and display the structure of the datasets."""
    print("\n" + "=" * 80)
    print("STEP 2: Exploring Dataset Structure")
    print("=" * 80)

    print("\n📊 Metadata (info) columns:")
    for col in info_df.columns:
        print(f"  - {col}")

    print("\n📊 Text dataset columns:")
    for col in text_df.columns:
        print(f"  - {col}")

    print("\n📖 Sample metadata entry:")
    print(info_df.head(1).to_dict("records")[0])

    print("\n📖 Sample text entry (first 200 chars):")
    sample = text_df.head(1).to_dict("records")[0]
    for key, value in sample.items():
        if isinstance(value, str) and len(value) > 200:
            print(f"  {key}: {value[:200]}...")
        else:
            print(f"  {key}: {value}")


def filter_books(info_df, text_df, selected_books):
    """Filter datasets to include only the 63 selected books."""
    print("\n" + "=" * 80)
    print("STEP 3: Filtering for Selected Books")
    print("=" * 80)

    # Determine which column contains book titles
    # Common column names: 'title', 'name', 'book_name', 'book_title', etc.
    title_column = None
    for col in info_df.columns:
        if any(keyword in col.lower() for keyword in ["title", "name", "book"]):
            title_column = col
            break

    if title_column is None:
        print("⚠️  Could not automatically detect title column. Using first column.")
        title_column = info_df.columns[0]

    print(f"\n🔍 Using column '{title_column}' for book titles")

    # Filter using partial matching (since titles might not be exact)
    matched_info = []
    matched_books = []

    for book_title in selected_books:
        # Try to find matches
        mask = (
            info_df[title_column]
            .astype(str)
            .str.contains(book_title, na=False, regex=False)
        )
        matches = info_df[mask]

        if len(matches) > 0:
            matched_info.append(matches.iloc[0])
            matched_books.append(book_title)
            print(f"✓ Found: {book_title}")
        else:
            print(f"✗ Not found: {book_title}")

    filtered_info_df = pd.DataFrame(matched_info)

    print(f"\n📊 Summary:")
    print(f"  Requested: {len(selected_books)} books")
    print(f"  Found: {len(filtered_info_df)} books")
    print(f"  Missing: {len(selected_books) - len(filtered_info_df)} books")

    return filtered_info_df, title_column


def merge_text_data(filtered_info_df, text_df):
    """Merge filtered metadata with full text data."""
    print("\n" + "=" * 80)
    print("STEP 4: Merging Metadata with Full Text")
    print("=" * 80)

    # Try to find common ID column
    common_cols = set(filtered_info_df.columns) & set(text_df.columns)
    print(f"\n🔗 Common columns for merging: {common_cols}")

    # Use 'id' or 'book_id' if available
    merge_col = None
    for col in ["id", "book_id", "bookId", "ID"]:
        if col in common_cols:
            merge_col = col
            break

    if merge_col:
        print(f"✓ Merging on column: {merge_col}")
        merged_df = pd.merge(filtered_info_df, text_df, on=merge_col, how="inner")
    else:
        print("⚠️  No common ID column found. Using index-based approach.")
        merged_df = filtered_info_df.copy()

    print(f"✓ Merged dataset contains {len(merged_df)} books")

    return merged_df


def save_results(merged_df, title_column):
    """Save the filtered data to CSV and individual markdown files."""
    print("\n" + "=" * 80)
    print("STEP 5: Saving Results")
    print("=" * 80)

    # Create output directories
    output_dir = Path("output")
    books_dir = output_dir / "books"
    output_dir.mkdir(exist_ok=True)
    books_dir.mkdir(exist_ok=True)

    # Save master CSV
    csv_path = output_dir / "selected_books.csv"
    merged_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n✓ Saved metadata CSV: {csv_path}")

    # Save individual books as markdown files
    text_column = None
    for col in merged_df.columns:
        if "text" in col.lower() or "content" in col.lower():
            text_column = col
            break

    if text_column:
        print(f"\n📝 Saving individual books (using column: {text_column})...")
        for idx, row in merged_df.iterrows():
            title = row[title_column]
            # Clean filename
            safe_title = "".join(
                c for c in title if c.isalnum() or c in (" ", "-", "_")
            ).strip()
            safe_title = safe_title.replace(" ", "_")[:100]  # Limit length

            book_path = books_dir / f"{idx + 1:02d}_{safe_title}.md"

            with open(book_path, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n")
                f.write(f"**Book ID:** {idx + 1}\n\n")
                f.write("---\n\n")
                f.write(row[text_column])

            print(f"  ✓ Saved: {book_path.name}")
    else:
        print("⚠️  No text column found in merged data")

    print(f"\n✅ All results saved to: {output_dir}/")


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("SHAMELA BOOK EXTRACTOR")
    print("Extracting 63 Authoritative Islamic Texts")
    print("=" * 80)

    try:
        # Download datasets
        info_df, text_df = download_datasets()

        # Explore structure
        explore_structure(info_df, text_df)

        # Filter for selected books
        filtered_info_df, title_column = filter_books(info_df, text_df, SELECTED_BOOKS)

        # Merge with text data
        merged_df = merge_text_data(filtered_info_df, text_df)

        # Save results
        save_results(merged_df, title_column)

        print("\n" + "=" * 80)
        print("✅ EXTRACTION COMPLETE!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Review output/selected_books.csv for metadata")
        print("  2. Check output/books/ for individual book files")
        print("  3. Verify that all 63 books were found")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
