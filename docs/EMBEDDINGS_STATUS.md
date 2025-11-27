# Embedding Status – Shamela Top Subset

_Last updated: 2025-02-16_

## Scope

We trimmed the original 65-book corpus down to 17 high-priority titles across Tafsir, Hadith, Fiqh/Usul, and Tarikh. For each title we:

1. Cleaned and chunked the assembled text (`output/clean_books/*.txt`, `output/chunks/*.jsonl`).
2. Combined per-book chunk JSONLs only when needed (`output/chunks/all_chunks.jsonl`).
3. Generated standalone embedding shards (`output/embeddings/per_book/{book_id}.parquet`) plus per-book summaries.

An aborted attempt to embed the entire combined JSONL produced `output/embeddings/all_chunks_embeddings.parquet`; it can be discarded safely because the per-book shards are authoritative.

## Embedded Books (17 / 17)

| Book ID | Collection / Type | Title | Chunk Count |
| --- | --- | --- | ---: |
| 22669 | Biographical | سير أعلام النبلاء | 16,410 |
| 5423 | Fiqh (Hanafi) | المبسوط | 12,607 |
| 6910 | Fiqh (Hanbali) | المغني | 15,710 |
| 587 | Fiqh (Maliki) | المدونة | 4,857 |
| 1655 | Fiqh (Shafi'i) | الأم | 4,420 |
| 28107 | Hadith | الموطأ | 3,442 |
| 117359 | Hadith | سنن أبي داود | 8,274 |
| 1435 | Hadith | سنن الترمذي | 3,893 |
| 735 | Hadith | صحيح البخاري | 7,566 |
| 1727 | Hadith | صحيح مسلم | 5,785 |
| 1711 | Hadith Commentary | المنهاج شرح صحيح مسلم بن الحجاج | 6,387 |
| 9242 | Hadith Commentary | نيل الأوطار | 6,153 |
| 20855 | Tafsir | الجامع لأحكام القرآن | 14,922 |
| 43 | Tafsir | جامع البيان عن تأويل آي القرآن | 26,422 |
| 4445 | Tarikh | البداية والنهاية | 24,076 |
| 8180 | Usul al-Fiqh | الرسالة | 1,217 |
| 11435 | Usul al-Fiqh | الموافقات | 5,737 |

All summaries live next to their Parquet siblings as `{book_id}_summary.json` and include tokenizer stats (mean norm, total characters, etc.).

## What Worked Well

- Per-book embedding runs: each Parquet shard is restart-safe, easy to ingest, and small enough (<2 GB each) not to exhaust disk/RAM.
- `combine_chunks.py` works cleanly on the reduced set (434 MB combined).
- `embed_chunks.py`’s validation catches empty text and missing IDs, so we have zero skipped chunks.

## What Failed / Lessons

- Running `embed_chunks.py` against the entire `all_chunks.jsonl` on a 16 GB RAM CPU box takes too long and risks disk exhaustion; stick with per-book or small shards.
- `clean_books.py --limit` requires an integer argument; we now just let it scan all files (only 17 remain).
- Attempting to resume an interrupted `python embed_chunks.py ...` doesn’t work; each run must start from the beginning of whichever file it targets.

## Next Steps

1. **Dependencies**  
   `pyproject.toml` now includes `pandas`, `pyarrow`, `tqdm`, and `sentence-transformers`. Mirror any future dependency tweaks in `requirements.txt` and `pyproject.toml`.

2. **Chunk QA**  
   Enhance `combine_chunks.py` stats (e.g., duplicate chunk detection, missing `[PAGE]` markers) to flag cleanup opportunities before embedding.

3. **Spot-check embeddings**  
   For a few books (one per category), open the Parquet files, ensure metadata fields (`collection_label`, `page_start/page_end`, etc.) survived intact, and verify chunk text/overlap.

4. **Vector ingestion**  
   - Option A: load `output/embeddings/per_book/*.parquet` straight into your vector store, then delete locally to reclaim space.
   - Option B: concatenate selected Parquets into a single `top_subset_embeddings.parquet` once you have enough disk headroom.

5. **Documentation & automation**  
   - Add a `book_selection.md` (or similar) describing the allow-list used to trim to 17 books.
   - Consider scripting the per-book embedding loop so you can rerun it with `make embed` or a simple shell wrapper.

## Housekeeping

- Remove `output/embeddings/all_chunks_embeddings.parquet` once you confirm all per-book Parquets are backed up or ingested.
- Keep `output/tmp/target_book_ids.txt` (or equivalent) in version control so future runs know which titles stay in scope.

Let me know when you’re ready to expand the subset, add chunk diagnostics, or build the ingestion harness.
