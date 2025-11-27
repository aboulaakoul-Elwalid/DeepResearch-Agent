#!/usr/bin/env python3
"""
Assemble Shamela Book Pages into Complete Books

This script:
1. Reads the downloaded parquet files from datasets/shamela_text/
2. Filters for the 65 target books
3. Groups pages by book_id
4. Sorts pages by volume_number and page_number
5. Concatenates all pages into complete books
6. Saves as clean markdown files

Usage:
    python assemble_books.py
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm


def load_matched_books() -> Dict[str, Dict]:
    """Load the matched book IDs from metadata."""
    with open("output/metadata/matched_books.json", "r", encoding="utf-8") as f:
        matched_books = json.load(f)

    # Create a set of book IDs for fast lookup
    target_ids = {str(book_info["id"]) for book_info in matched_books.values()}

    return matched_books, target_ids


def parse_volume_number(vol_str: str) -> Tuple[int, int]:
    """
    Parse volume number string into (volume, part) tuple for sorting.

    Examples:
        "1" -> (1, 0)
        "5 أ" -> (5, 1)
        "5 ب" -> (5, 2)
        "2-1" -> (2, 1)
        "47 - 48" -> (47, 0)
    """
    if pd.isna(vol_str) or vol_str == "" or vol_str == "0":
        return (0, 0)

    vol_str = str(vol_str).strip()

    # Handle hyphenated parts "2-1"
    if "-" in vol_str:
        parts = vol_str.split("-")
        try:
            return (
                int(parts[0].strip()),
                int(parts[1].strip()) if len(parts) > 1 else 0,
            )
        except:
            return (0, 0)

    # Handle Arabic letter parts "5 أ"
    if " أ" in vol_str:
        vol = vol_str.replace(" أ", "").strip()
        try:
            return (int(vol), 1)
        except:
            return (0, 1)

    if " ب" in vol_str:
        vol = vol_str.replace(" ب", "").strip()
        try:
            return (int(vol), 2)
        except:
            return (0, 2)

    # Try direct conversion
    try:
        return (int(vol_str), 0)
    except:
        return (0, 0)


def parse_page_number(page_str: str) -> int:
    """Parse page number string into integer."""
    if pd.isna(page_str) or page_str == "":
        return 0

    try:
        return int(str(page_str).strip())
    except:
        return 0


def load_pages_from_parquet(target_ids: set) -> Dict[str, List[Dict]]:
    """
    Load all pages for target books from parquet files.
    Returns a dict: {book_id: [list of page dicts]}
    """
    print("\n" + "=" * 80)
    print("STEP 1: Loading Pages from Parquet Files")
    print("=" * 80)

    parquet_dir = Path("datasets/shamela_text/data")
    parquet_files = sorted(parquet_dir.glob("train-*.parquet"))

    print(f"📂 Found {len(parquet_files)} parquet files")

    # Dictionary to accumulate pages per book
    book_pages = defaultdict(list)

    # Process each parquet file
    for parquet_file in tqdm(parquet_files, desc="Reading parquet files"):
        # Read the parquet file
        df = pd.read_parquet(parquet_file)

        # Filter for target books only
        mask = df["book_id"].astype(str).isin(target_ids)
        filtered_df = df[mask]

        if len(filtered_df) == 0:
            continue

        # Add pages to book_pages dictionary
        for _, row in filtered_df.iterrows():
            book_id = str(row["book_id"])
            book_pages[book_id].append(
                {
                    "book_id": book_id,
                    "book_title": row.get("book_title", ""),
                    "volume_number": row.get("volume_number", ""),
                    "page_number": row.get("page_number", ""),
                    "text": row.get("text", ""),
                    "foot_note": row.get("foot_note", ""),
                    "category": row.get("category", ""),
                }
            )

    print(f"\n✓ Loaded pages for {len(book_pages)} books")

    # Print summary
    for book_id, pages in book_pages.items():
        print(f"  Book ID {book_id}: {len(pages)} pages")

    return dict(book_pages)


def sort_pages(pages: List[Dict]) -> List[Dict]:
    """Sort pages by volume number and page number."""

    def sort_key(page):
        vol_tuple = parse_volume_number(page["volume_number"])
        page_num = parse_page_number(page["page_number"])
        return (vol_tuple[0], vol_tuple[1], page_num)

    return sorted(pages, key=sort_key)


def assemble_book(pages: List[Dict], metadata: Dict) -> str:
    """
    Assemble all pages into a single markdown document.

    Returns the complete book text with proper formatting.
    """
    # Sort pages
    sorted_pages = sort_pages(pages)

    # Build the book content
    lines = []

    # Header
    lines.append(f"# {metadata['title']}")
    lines.append("")
    lines.append(f"**Book ID:** {metadata['id']}")
    lines.append(f"**Total Pages:** {len(sorted_pages)}")

    # Add metadata if available
    if metadata.get("metadata"):
        meta = metadata["metadata"]
        if meta.get("author_name"):
            lines.append(f"**Author:** {meta['author_name']}")
        if meta.get("category"):
            lines.append(f"**Category:** {meta['category']}")
        if meta.get("publisher"):
            lines.append(f"**Publisher:** {meta['publisher']}")
        if meta.get("volumes"):
            lines.append(f"**Volumes:** {meta['volumes']}")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Content - organized by volume
    current_volume = None

    for page in sorted_pages:
        vol_str = page["volume_number"]
        page_num = page["page_number"]

        # Add volume header if changed
        if vol_str != current_volume and vol_str:
            current_volume = vol_str
            lines.append("")
            lines.append(f"## Volume {vol_str}")
            lines.append("")

        # Add page marker (optional, can be commented out for cleaner text)
        if page_num:
            lines.append(f"<!-- Page {page_num} -->")

        # Add main text
        text = str(page["text"]).strip()
        if text:
            lines.append(text)
            lines.append("")

        # Add footnote if exists
        footnote = str(page["foot_note"]).strip()
        if footnote and footnote != "nan":
            lines.append(f"> **Footnote:** {footnote}")
            lines.append("")

    return "\n".join(lines)


def save_assembled_books(book_pages: Dict[str, List[Dict]], matched_books: Dict):
    """Save all assembled books as markdown files."""
    print("\n" + "=" * 80)
    print("STEP 2: Assembling and Saving Books")
    print("=" * 80)

    # Create output directory
    assembled_dir = Path("output/assembled_books")
    assembled_dir.mkdir(parents=True, exist_ok=True)

    # Clear old single-page files
    old_books_dir = Path("output/books")
    if old_books_dir.exists():
        print("🗑️  Cleaning up old single-page files...")
        for old_file in old_books_dir.glob("*.md"):
            old_file.unlink()

    # Assemble each book
    saved_count = 0

    for book_id, pages in tqdm(book_pages.items(), desc="Assembling books"):
        # Find metadata
        metadata = None
        for book_info in matched_books.values():
            if str(book_info["id"]) == book_id:
                metadata = book_info
                break

        if not metadata:
            print(f"⚠️  No metadata found for book_id {book_id}, skipping...")
            continue

        # Assemble the book
        book_content = assemble_book(pages, metadata)

        # Create safe filename
        title = metadata["title"]
        safe_title = "".join(
            c for c in title if c.isalnum() or c in (" ", "-", "_", "،", "؛")
        ).strip()
        safe_title = safe_title.replace(" ", "_")[:100]

        # Save
        output_path = assembled_dir / f"{book_id}_{safe_title}.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(book_content)

        saved_count += 1

        # Print sample info
        if saved_count <= 5:
            print(f"✓ Saved: {title} ({len(pages)} pages)")

    print(f"\n✅ Successfully assembled {saved_count} books")
    print(f"📁 Output directory: {assembled_dir}")


def create_book_index(book_pages: Dict[str, List[Dict]], matched_books: Dict):
    """Create an index CSV with book information."""
    print("\n" + "=" * 80)
    print("STEP 3: Creating Book Index")
    print("=" * 80)

    records = []

    for book_id, pages in book_pages.items():
        # Find metadata
        metadata = None
        for book_info in matched_books.values():
            if str(book_info["id"]) == book_id:
                metadata = book_info
                break

        if not metadata:
            continue

        # Count volumes
        volumes = set(page["volume_number"] for page in pages if page["volume_number"])

        record = {
            "book_id": book_id,
            "title": metadata["title"],
            "total_pages": len(pages),
            "total_volumes": len(volumes),
            "category": pages[0].get("category", "") if pages else "",
            "file_path": f"output/assembled_books/{book_id}_{metadata['title'][:50]}.md",
        }

        # Add author info if available
        if metadata.get("metadata"):
            meta = metadata["metadata"]
            record["author"] = meta.get("author_name", "")
            record["publisher"] = meta.get("publisher", "")

        records.append(record)

    # Save to CSV
    df = pd.DataFrame(records)
    df = df.sort_values("title")

    index_path = "output/assembled_books_index.csv"
    df.to_csv(index_path, index=False, encoding="utf-8")

    print(f"✓ Saved index: {index_path}")
    print(f"\n📊 Summary by Category:")
    print(df.groupby("category")["book_id"].count())


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("SHAMELA BOOK ASSEMBLER")
    print("Combining Pages into Complete Books")
    print("=" * 80)

    try:
        # Load matched book IDs
        matched_books, target_ids = load_matched_books()
        print(f"\n🎯 Target: {len(target_ids)} books")

        # Load all pages from parquet files
        book_pages = load_pages_from_parquet(target_ids)

        if not book_pages:
            print(
                "\n❌ No pages found! Check that parquet files exist in datasets/shamela_text/data/"
            )
            return

        # Assemble and save books
        save_assembled_books(book_pages, matched_books)

        # Create index
        create_book_index(book_pages, matched_books)

        print("\n" + "=" * 80)
        print("✅ ASSEMBLY COMPLETE!")
        print("=" * 80)
        print("\n📁 Your complete books are in: output/assembled_books/")
        print("📊 Index file: output/assembled_books_index.csv")
        print("\n💡 Next steps:")
        print("   - Review the assembled books")
        print("   - Process them for embeddings")
        print("   - Create chunking strategy for RAG")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
