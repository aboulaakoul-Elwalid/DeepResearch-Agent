#!/bin/bash
#
# RAG Pipeline Orchestration Script
# 
# This script orchestrates the complete RAG pipeline from extraction to ChromaDB ingestion.
# It can run the entire pipeline or individual stages.
#
# Usage:
#   ./run_pipeline.sh                    # Run complete pipeline
#   ./run_pipeline.sh extract            # Run only extraction
#   ./run_pipeline.sh chunk embed ingest # Run specific stages
#   ./run_pipeline.sh --dry-run          # Show what would be run
#   ./run_pipeline.sh --help             # Show help
#

set -e  # Exit on error

# ============================================================================
# Configuration (can be overridden via environment variables)
# ============================================================================

# Directories
OUTPUT_DIR="${OUTPUT_DIR:-output}"
BOOKS_DIR="${OUTPUT_DIR}/books"
CLEAN_DIR="${OUTPUT_DIR}/clean_books"
MANIFEST_DIR="${OUTPUT_DIR}/manifest"
CHUNKS_DIR="${OUTPUT_DIR}/chunks"
EMBEDDINGS_DIR="${OUTPUT_DIR}/embeddings"
CHROMA_DIR="${CHROMA_DIR:-chroma_db}"

# Processing parameters
CHUNK_SIZE="${CHUNK_SIZE:-1200}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo "========================================================================"
    echo " $1"
    echo "========================================================================"
}

print_step() {
    echo ""
    echo ">>> $1"
}

print_success() {
    echo "[OK] $1"
}

print_warning() {
    echo "[WARN] $1"
}

print_error() {
    echo "[ERROR] $1" >&2
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not found"
        exit 1
    fi
}

show_help() {
    cat << EOF
RAG Pipeline Orchestration Script

USAGE:
    ./run_pipeline.sh [OPTIONS] [STAGES...]

STAGES:
    extract     - Extract books from HuggingFace dataset (extract_books_smart.py or extract_via_api.py)
    assemble    - Assemble extracted book parts (assemble_books.py)
    clean       - Clean and normalize text (clean_books.py)
    manifest    - Build metadata manifest (build_manifest.py)
    chunk       - Create text chunks (create_chunks.py)
    combine     - Combine chunks into single file (combine_chunks.py)
    embed       - Generate embeddings (embed_chunks.py or embed_chunks_per_book.py)
    qa          - Spot-check embeddings (spot_check_embeddings.py)
    ingest      - Ingest into ChromaDB (ingest_chroma.py)
    
    If no stages are specified, all stages are run in order.

OPTIONS:
    --dry-run       Show what would be run without executing
    --verbose       Enable verbose output
    --limit N       Limit processing to N books (for testing)
    --skip-existing Skip stages that have already produced output
    --per-book      Use per-book embedding strategy (for ingest_chroma.py)
    --help          Show this help message

ENVIRONMENT VARIABLES:
    OUTPUT_DIR          Base output directory (default: output)
    CHUNK_SIZE          Target chunk size in chars (default: 1200)
    CHUNK_OVERLAP       Overlap between chunks (default: 200)
    BATCH_SIZE          Embedding batch size (default: 64)
    EMBEDDING_MODEL     SentenceTransformer model (default: paraphrase-multilingual-MiniLM-L12-v2)
    CHROMA_DIR          ChromaDB storage path (default: chroma_db)

EXAMPLES:
    # Run complete pipeline
    ./run_pipeline.sh

    # Run only chunking and embedding
    ./run_pipeline.sh chunk embed

    # Dry run to see what would happen
    ./run_pipeline.sh --dry-run

    # Process only 5 books for testing
    ./run_pipeline.sh --limit 5

    # Run with custom chunk size
    CHUNK_SIZE=1500 ./run_pipeline.sh chunk

EOF
}

# ============================================================================
# Stage Functions
# ============================================================================

stage_extract() {
    print_step "Stage: Extract books from HuggingFace"
    
    if [[ "$SKIP_EXISTING" == "true" ]] && [[ -d "$BOOKS_DIR" ]] && [[ -n "$(ls -A "$BOOKS_DIR" 2>/dev/null)" ]]; then
        print_warning "Skipping extraction - books already exist in $BOOKS_DIR"
        return 0
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  Would run: python3 $SCRIPT_DIR/extract_books_smart.py"
        return 0
    fi
    
    # Prefer smart extraction, fall back to API extraction
    if [[ -f "$SCRIPT_DIR/extract_books_smart.py" ]]; then
        python3 "$SCRIPT_DIR/extract_books_smart.py" ${LIMIT:+--limit $LIMIT} ${VERBOSE:+--verbose}
    elif [[ -f "$SCRIPT_DIR/extract_via_api.py" ]]; then
        python3 "$SCRIPT_DIR/extract_via_api.py"
    else
        print_error "No extraction script found"
        return 1
    fi
    
    print_success "Extraction complete"
}

stage_assemble() {
    print_step "Stage: Assemble book parts"
    
    if [[ "$SKIP_EXISTING" == "true" ]] && [[ -d "$BOOKS_DIR" ]] && [[ -n "$(find "$BOOKS_DIR" -name "*.txt" 2>/dev/null | head -1)" ]]; then
        print_warning "Skipping assembly - assembled books already exist"
        return 0
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  Would run: python3 $SCRIPT_DIR/assemble_books.py"
        return 0
    fi
    
    python3 "$SCRIPT_DIR/assemble_books.py" ${VERBOSE:+--verbose}
    print_success "Assembly complete"
}

stage_clean() {
    print_step "Stage: Clean and normalize text"
    
    if [[ "$SKIP_EXISTING" == "true" ]] && [[ -d "$CLEAN_DIR" ]] && [[ -n "$(ls -A "$CLEAN_DIR" 2>/dev/null)" ]]; then
        print_warning "Skipping cleaning - clean books already exist in $CLEAN_DIR"
        return 0
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  Would run: python3 $SCRIPT_DIR/clean_books.py --output-dir $CLEAN_DIR"
        return 0
    fi
    
    python3 "$SCRIPT_DIR/clean_books.py" \
        --output-dir "$CLEAN_DIR" \
        ${LIMIT:+--limit $LIMIT} \
        ${VERBOSE:+--verbose}
    
    print_success "Cleaning complete"
}

stage_manifest() {
    print_step "Stage: Build metadata manifest"
    
    local manifest_file="$MANIFEST_DIR/books_manifest.csv"
    
    if [[ "$SKIP_EXISTING" == "true" ]] && [[ -f "$manifest_file" ]]; then
        print_warning "Skipping manifest - already exists at $manifest_file"
        return 0
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  Would run: python3 $SCRIPT_DIR/build_manifest.py --output $manifest_file"
        return 0
    fi
    
    mkdir -p "$MANIFEST_DIR"
    python3 "$SCRIPT_DIR/build_manifest.py" \
        --output "$manifest_file" \
        ${VERBOSE:+--verbose}
    
    print_success "Manifest created at $manifest_file"
}

stage_chunk() {
    print_step "Stage: Create text chunks"
    
    if [[ "$SKIP_EXISTING" == "true" ]] && [[ -d "$CHUNKS_DIR" ]] && [[ -n "$(ls -A "$CHUNKS_DIR" 2>/dev/null)" ]]; then
        print_warning "Skipping chunking - chunks already exist in $CHUNKS_DIR"
        return 0
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  Would run: python3 $SCRIPT_DIR/create_chunks.py --chunk-size $CHUNK_SIZE --overlap $CHUNK_OVERLAP"
        return 0
    fi
    
    python3 "$SCRIPT_DIR/create_chunks.py" \
        --manifest "$MANIFEST_DIR/books_manifest.csv" \
        --input-dir "$CLEAN_DIR" \
        --output-dir "$CHUNKS_DIR" \
        --chunk-size "$CHUNK_SIZE" \
        --overlap "$CHUNK_OVERLAP" \
        ${LIMIT:+--limit $LIMIT} \
        ${VERBOSE:+--verbose}
    
    print_success "Chunking complete"
}

stage_combine() {
    print_step "Stage: Combine chunks into single file"
    
    local combined_file="$CHUNKS_DIR/all_chunks.jsonl"
    
    if [[ "$SKIP_EXISTING" == "true" ]] && [[ -f "$combined_file" ]]; then
        print_warning "Skipping combine - $combined_file already exists"
        return 0
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  Would run: python3 $SCRIPT_DIR/combine_chunks.py --output $combined_file"
        return 0
    fi
    
    python3 "$SCRIPT_DIR/combine_chunks.py" \
        --input-dir "$CHUNKS_DIR" \
        --output "$combined_file" \
        ${VERBOSE:+--verbose}
    
    print_success "Combined chunks at $combined_file"
}

stage_embed() {
    print_step "Stage: Generate embeddings"
    
    mkdir -p "$EMBEDDINGS_DIR"
    
    if [[ "$USE_PER_BOOK" == "true" ]]; then
        # Per-book embeddings for ingest_chroma.py
        local per_book_dir="$EMBEDDINGS_DIR/per_book"
        
        if [[ "$SKIP_EXISTING" == "true" ]] && [[ -d "$per_book_dir" ]] && [[ -n "$(ls -A "$per_book_dir" 2>/dev/null)" ]]; then
            print_warning "Skipping per-book embedding - files already exist in $per_book_dir"
            return 0
        fi
        
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "  Would run: python3 $SCRIPT_DIR/embed_chunks_per_book.py"
            return 0
        fi
        
        if [[ -f "$SCRIPT_DIR/embed_chunks_per_book.py" ]]; then
            python3 "$SCRIPT_DIR/embed_chunks_per_book.py" \
                --chunks-dir "$CHUNKS_DIR" \
                --output-dir "$per_book_dir" \
                --model "$EMBEDDING_MODEL" \
                --batch-size "$BATCH_SIZE" \
                ${VERBOSE:+--verbose}
        else
            print_error "embed_chunks_per_book.py not found"
            return 1
        fi
    else
        # Monolithic embeddings
        local parquet_file="$EMBEDDINGS_DIR/all_chunks_embeddings.parquet"
        
        if [[ "$SKIP_EXISTING" == "true" ]] && [[ -f "$parquet_file" ]]; then
            print_warning "Skipping embedding - $parquet_file already exists"
            return 0
        fi
        
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "  Would run: python3 $SCRIPT_DIR/embed_chunks.py --parquet-file $parquet_file"
            return 0
        fi
        
        python3 "$SCRIPT_DIR/embed_chunks.py" \
            --chunks-file "$CHUNKS_DIR/all_chunks.jsonl" \
            --parquet-file "$parquet_file" \
            --summary-file "$EMBEDDINGS_DIR/embedding_summary.json" \
            --model "$EMBEDDING_MODEL" \
            --batch-size "$BATCH_SIZE" \
            --normalize \
            ${LIMIT:+--limit $LIMIT} \
            ${VERBOSE:+--verbose}
    fi
    
    print_success "Embedding complete"
}

stage_qa() {
    print_step "Stage: QA spot-check embeddings"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  Would run: python3 $SCRIPT_DIR/spot_check_embeddings.py"
        return 0
    fi
    
    if [[ -f "$SCRIPT_DIR/spot_check_embeddings.py" ]]; then
        python3 "$SCRIPT_DIR/spot_check_embeddings.py" \
            --parquet "$EMBEDDINGS_DIR/all_chunks_embeddings.parquet" \
            ${VERBOSE:+--verbose} || true  # Don't fail pipeline on QA issues
    else
        print_warning "spot_check_embeddings.py not found, skipping QA"
    fi
}

stage_ingest() {
    print_step "Stage: Ingest into ChromaDB"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  Would run: python3 $SCRIPT_DIR/ingest_chroma.py --chroma-path $CHROMA_DIR"
        return 0
    fi
    
    local input_path
    if [[ "$USE_PER_BOOK" == "true" ]] && [[ -d "$EMBEDDINGS_DIR/per_book" ]]; then
        input_path="$EMBEDDINGS_DIR/per_book"
    else
        input_path="$EMBEDDINGS_DIR"
    fi
    
    python3 "$SCRIPT_DIR/ingest_chroma.py" \
        --inputs "$input_path" \
        --chroma-path "$CHROMA_DIR" \
        --collection "arabic_books" \
        --batch-size 750 \
        ${VERBOSE:+--log-level DEBUG}
    
    print_success "ChromaDB ingestion complete"
}

# ============================================================================
# Main
# ============================================================================

main() {
    # Parse arguments
    DRY_RUN=false
    VERBOSE=""
    LIMIT=""
    SKIP_EXISTING=false
    USE_PER_BOOK=false
    STAGES=()
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose)
                VERBOSE="--verbose"
                shift
                ;;
            --limit)
                LIMIT="$2"
                shift 2
                ;;
            --skip-existing)
                SKIP_EXISTING=true
                shift
                ;;
            --per-book)
                USE_PER_BOOK=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            extract|assemble|clean|manifest|chunk|combine|embed|qa|ingest)
                STAGES+=("$1")
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Default to all stages if none specified
    if [[ ${#STAGES[@]} -eq 0 ]]; then
        STAGES=(extract assemble clean manifest chunk combine embed qa ingest)
    fi
    
    # Check prerequisites
    check_python
    
    # Print configuration
    print_header "RAG Pipeline"
    echo "Configuration:"
    echo "  Output directory: $OUTPUT_DIR"
    echo "  Chunk size: $CHUNK_SIZE chars"
    echo "  Chunk overlap: $CHUNK_OVERLAP chars"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Embedding model: $EMBEDDING_MODEL"
    echo "  ChromaDB path: $CHROMA_DIR"
    echo "  Stages: ${STAGES[*]}"
    echo "  Dry run: $DRY_RUN"
    echo "  Per-book mode: $USE_PER_BOOK"
    [[ -n "$LIMIT" ]] && echo "  Limit: $LIMIT books"
    [[ "$SKIP_EXISTING" == "true" ]] && echo "  Skip existing: enabled"
    
    # Create output directories
    if [[ "$DRY_RUN" != "true" ]]; then
        mkdir -p "$OUTPUT_DIR" "$BOOKS_DIR" "$CLEAN_DIR" "$MANIFEST_DIR" "$CHUNKS_DIR" "$EMBEDDINGS_DIR"
    fi
    
    # Run stages
    for stage in "${STAGES[@]}"; do
        case "$stage" in
            extract)  stage_extract ;;
            assemble) stage_assemble ;;
            clean)    stage_clean ;;
            manifest) stage_manifest ;;
            chunk)    stage_chunk ;;
            combine)  stage_combine ;;
            embed)    stage_embed ;;
            qa)       stage_qa ;;
            ingest)   stage_ingest ;;
            *)
                print_error "Unknown stage: $stage"
                exit 1
                ;;
        esac
    done
    
    print_header "Pipeline Complete"
    echo "All requested stages completed successfully."
}

# Run main function
main "$@"
