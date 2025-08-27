#!/usr/bin/env python3
"""
RAG Script: Company Policies Chunking & Vectorization
Processes company policies text into chunks and generates embeddings for Pinecone upsert.
"""

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

import typer
import tiktoken
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

try:
    import orjson
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False

# Load environment variables
load_dotenv()

# Constants
TOKYO_TZ = timezone(timedelta(hours=9))
DEFAULT_MODEL = "text-embedding-3-small"
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

app = typer.Typer(help="RAG Script for Company Policies Processing")

# Logging setup
def setup_logging(verbose: bool = False):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

@dataclass
class Section:
    emoji: str
    title: str
    content: str
    page: int
    start_char: int
    end_char: int

@dataclass
class Chunk:
    text: str
    section: Section
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int

@dataclass
class EmbeddingRecord:
    id: str
    text: str
    vector: List[float]
    metadata: Dict[str, Any]

@dataclass
class ProcessingReport:
    total_chunks: int
    avg_token_count: float
    max_token_count: int
    avg_char_count: float
    max_char_count: int
    failed_chunks: int
    retry_count: int
    processing_time_seconds: float
    model_name: str
    input_path: str
    output_path: str
    section_distribution: Dict[str, int]
    vector_dimensions: int

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    return len(tokenizer.encode(text))

def normalize_text(text: str) -> str:
    """Normalize text by reducing multiple newlines to single ones."""
    # Replace multiple consecutive newlines with single newline
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()

def extract_title(text: str) -> Tuple[str, str]:
    """Extract title (first non-empty line) and return title and remaining text."""
    lines = text.strip().split('\n')
    if not lines:
        return "", text
    
    title = lines[0].strip()
    remaining_text = '\n'.join(lines[1:])
    return title, remaining_text

def parse_sections(text: str) -> List[Section]:
    """Parse sections from text based on emoji + space + title pattern."""
    sections = []
    
    # Pattern to match emoji + space + section title, allowing leading whitespace
    # Must NOT start with '-' (bullet points)
    section_pattern = r'^\s*([^\w\s\-])\s+(.+)$'
    
    lines = text.split('\n')
    current_section = None
    current_content_lines = []
    page_number = 1
    char_offset = 0
    
    for line in lines:
        original_line = line
        line = line.strip()
        
        # Skip empty lines
        if not line:
            char_offset += len(original_line) + 1
            continue
        
        # Check if this line is a section header (emoji + space + title, NOT bullet points)
        match = re.match(section_pattern, line)
        if match and not line.startswith('-'):
            # Save previous section if exists
            if current_section is not None:
                content = '\n'.join(current_content_lines).strip()
                if content:  # Only save sections with content
                    end_char = current_section['start_char'] + len(content)
                    sections.append(Section(
                        emoji=current_section['emoji'],
                        title=current_section['title'],
                        content=content,
                        page=page_number,
                        start_char=current_section['start_char'],
                        end_char=end_char
                    ))
                    page_number += 1
            
            # Start new section
            emoji, title = match.groups()
            current_section = {
                'emoji': emoji,
                'title': title,
                'start_char': char_offset
            }
            current_content_lines = []
        else:
            # Add line to current section content (including bullet points)
            if current_section is not None:
                current_content_lines.append(line)
        
        char_offset += len(original_line) + 1  # +1 for newline
    
    # Save final section
    if current_section is not None:
        content = '\n'.join(current_content_lines).strip()
        if content:  # Only save sections with content
            end_char = current_section['start_char'] + len(content)
            sections.append(Section(
                emoji=current_section['emoji'],
                title=current_section['title'],
                content=content,
                page=page_number,
                start_char=current_section['start_char'],
                end_char=end_char
            ))
    
    # If no sections found, treat entire text as single section
    if not sections and text.strip():
        sections.append(Section(
            emoji="📄",
            title="文書全体",
            content=text.strip(),
            page=1,
            start_char=0,
            end_char=len(text)
        ))
    
    return sections

def section_to_key(section_title: str) -> str:
    """Convert section title to English slug key."""
    title = section_title or ""
    if "勤務" in title or "休暇" in title:
        return "work_leave"
    if "IT" in title or "セキュリティ" in title:
        return "it_security"
    if "会議" in title or "備品" in title or "その他" in title:
        return "general_rules"
    return "unknown"

def split_at_boundaries(text: str, max_tokens: int) -> List[str]:
    """Split text at sentence/line boundaries without exceeding max_tokens."""
    if count_tokens(text) <= max_tokens:
        return [text]
    
    chunks = []
    
    # Split by sentences and line breaks, prioritizing bullet points
    sentences = re.split(r'。|\n+', text)
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Add period back if it was split by period
        if not sentence.endswith('。') and '。' in text:
            sentence_with_period = sentence + '。'
        else:
            sentence_with_period = sentence
        
        test_chunk = current_chunk + ("\n" if current_chunk else "") + sentence_with_period
        
        if count_tokens(test_chunk) <= max_tokens:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence_with_period
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return [chunk for chunk in chunks if chunk.strip()]

def add_overlap(chunks: List[str], overlap_tokens: int) -> List[str]:
    """Add overlap between consecutive chunks."""
    if len(chunks) <= 1 or overlap_tokens <= 0:
        return chunks
    
    overlapped_chunks = [chunks[0]]
    
    for i in range(1, len(chunks)):
        prev_chunk = chunks[i-1]
        current_chunk = chunks[i]
        
        # Get last part of previous chunk for overlap
        prev_tokens = tokenizer.encode(prev_chunk)
        if len(prev_tokens) >= overlap_tokens:
            overlap_tokens_actual = prev_tokens[-overlap_tokens:]
            overlap_text = tokenizer.decode(overlap_tokens_actual)
            
            combined_chunk = overlap_text + "\n" + current_chunk
            overlapped_chunks.append(combined_chunk)
        else:
            overlapped_chunks.append(current_chunk)
    
    return overlapped_chunks

def chunk_section(section: Section, target_tokens: int, overlap_tokens: int, 
                  min_chars: int, max_chars: int) -> List[Chunk]:
    """Chunk a section into smaller pieces."""
    chunks = []
    
    # Split section content into chunks
    raw_chunks = split_at_boundaries(section.content, target_tokens)
    
    # Add overlap between chunks
    overlapped_chunks = add_overlap(raw_chunks, overlap_tokens)
    
    # Apply character limits and create Chunk objects
    chunk_index = 0
    char_offset = section.start_char
    
    for chunk_text in overlapped_chunks:
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue
        
        # Keep chunks even if shorter than min_chars to avoid losing content
        # Only skip if completely empty
        
        if len(chunk_text) > max_chars:
            # Force split if too long
            sub_chunks = split_at_boundaries(chunk_text, target_tokens // 2)
            for sub_chunk in sub_chunks:
                if sub_chunk.strip():  # Only skip if empty
                    token_count = count_tokens(sub_chunk)
                    chunks.append(Chunk(
                        text=sub_chunk,
                        section=section,
                        chunk_index=chunk_index,
                        start_char=char_offset,
                        end_char=char_offset + len(sub_chunk),
                        token_count=token_count
                    ))
                    chunk_index += 1
                    char_offset += len(sub_chunk)
        else:
            token_count = count_tokens(chunk_text)
            chunks.append(Chunk(
                text=chunk_text,
                section=section,
                chunk_index=chunk_index,
                start_char=char_offset,
                end_char=char_offset + len(chunk_text),
                token_count=token_count
            ))
            chunk_index += 1
            char_offset += len(chunk_text)
    
    return chunks

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_embeddings(texts: List[str], model: str, client: OpenAI) -> List[List[float]]:
    """Get embeddings from OpenAI API with retry logic."""
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    
    # Verify response order matches input order
    if len(response.data) != len(texts):
        raise ValueError(f"Response count {len(response.data)} != input count {len(texts)}")
    
    return [data.embedding for data in response.data]

def embed_chunks(chunks: List[Chunk], model: str, batch_size: int, 
                dry_run: bool, logger: logging.Logger) -> Tuple[List[List[float]], int]:
    """Embed chunks in batches."""
    if dry_run:
        logger.info("Dry run mode: skipping embeddings API calls")
        return [[] for _ in chunks], 0
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    all_embeddings = []
    retry_count = 0
    
    logger.info(f"Embedding {len(chunks)} chunks with model {model} (batch size: {batch_size})")
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_texts = [chunk.text for chunk in batch_chunks]
        
        try:
            batch_embeddings = get_embeddings(batch_texts, model, client)
            all_embeddings.extend(batch_embeddings)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
        except Exception as e:
            logger.error(f"Failed to get embeddings for batch starting at {i}: {e}")
            retry_count += 1
            # Add empty embeddings as fallback
            all_embeddings.extend([[] for _ in batch_chunks])
    
    # Verify dimensions for non-empty vectors
    if not dry_run and all_embeddings:
        non_empty_embeddings = [emb for emb in all_embeddings if emb]
        if non_empty_embeddings:
            expected_dim = MODEL_DIMENSIONS.get(model, 1536)
            actual_dim = len(non_empty_embeddings[0])
            if actual_dim != expected_dim:
                logger.warning(f"Unexpected vector dimension: {actual_dim} (expected {expected_dim})")
    
    return all_embeddings, retry_count

def generate_record_id(namespace: str, section_key: str, page: int, seq: int) -> str:
    """Generate record ID following the specified format."""
    return f"{namespace}_s{section_key}_p{page}_c{seq:04d}"

def chunk_to_record(chunk: Chunk, embedding: List[float], namespace: str, 
                   title: str, seq: int) -> EmbeddingRecord:
    """Convert chunk to embedding record."""
    section_key = section_to_key(chunk.section.title)
    record_id = generate_record_id(namespace, section_key, chunk.section.page, seq)
    
    now = datetime.now(TOKYO_TZ)
    
    metadata = {
        "source": "company_policies_rag_dataset.txt",
        "title": title,
        "section": chunk.section.title,
        "section_emoji": chunk.section.emoji,
        "page": chunk.section.page,
        "chunk_index": chunk.chunk_index,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "token_count": chunk.token_count,
        "updated_at": now.isoformat(),
        "category": section_key
    }
    
    return EmbeddingRecord(
        id=record_id,
        text=chunk.text,
        vector=embedding,
        metadata=metadata
    )

def save_jsonl(records: List[EmbeddingRecord], output_path: Path, logger: logging.Logger):
    """Save records to JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            record_dict = asdict(record)
            
            if USE_ORJSON:
                json_str = orjson.dumps(record_dict, option=orjson.OPT_SERIALIZE_NUMPY).decode('utf-8')
            else:
                json_str = json.dumps(record_dict, ensure_ascii=False)
            
            f.write(json_str + '\n')
    
    logger.info(f"Saved {len(records)} records to {output_path}")

def generate_report(chunks: List[Chunk], embeddings: List[List[float]], 
                   retry_count: int, processing_time: float, model: str,
                   input_path: str, output_path: str) -> ProcessingReport:
    """Generate processing report."""
    if not chunks:
        return ProcessingReport(
            total_chunks=0, avg_token_count=0, max_token_count=0,
            avg_char_count=0, max_char_count=0, failed_chunks=0,
            retry_count=retry_count, processing_time_seconds=processing_time,
            model_name=model, input_path=input_path, output_path=output_path,
            section_distribution={}, vector_dimensions=0
        )
    
    token_counts = [chunk.token_count for chunk in chunks]
    char_counts = [len(chunk.text) for chunk in chunks]
    failed_chunks = sum(1 for emb in embeddings if not emb)
    
    # Section distribution
    section_dist = {}
    for chunk in chunks:
        section_key = section_to_key(chunk.section.title)
        section_dist[section_key] = section_dist.get(section_key, 0) + 1
    
    # Vector dimensions
    vector_dims = 0
    if embeddings:
        non_empty = [emb for emb in embeddings if emb]
        if non_empty:
            vector_dims = len(non_empty[0])
    
    return ProcessingReport(
        total_chunks=len(chunks),
        avg_token_count=sum(token_counts) / len(token_counts),
        max_token_count=max(token_counts),
        avg_char_count=sum(char_counts) / len(char_counts),
        max_char_count=max(char_counts),
        failed_chunks=failed_chunks,
        retry_count=retry_count,
        processing_time_seconds=processing_time,
        model_name=model,
        input_path=input_path,
        output_path=output_path,
        section_distribution=section_dist,
        vector_dimensions=vector_dims
    )

def save_report(report: ProcessingReport, report_path: Path, logger: logging.Logger):
    """Save processing report to JSON."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        if USE_ORJSON:
            json_str = orjson.dumps(asdict(report), option=orjson.OPT_INDENT_2).decode('utf-8')
            f.write(json_str)
        else:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved processing report to {report_path}")

@app.command()
def main(
    input: Path = typer.Option(..., "--input", help="Input text file path"),
    output: Path = typer.Option(..., "--output", help="Output JSONL file path"),
    report: Path = typer.Option("out/report.json", "--report", help="Report JSON file path"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="OpenAI embedding model"),
    target_tokens: int = typer.Option(300, "--target-tokens", help="Target tokens per chunk"),
    overlap_tokens: int = typer.Option(80, "--overlap-tokens", help="Overlap tokens between chunks"),
    min_chars: int = typer.Option(80, "--min-chars", help="Minimum characters per chunk"),
    max_chars: int = typer.Option(1800, "--max-chars", help="Maximum characters per chunk"),
    namespace: str = typer.Option("company-policies-v1", "--namespace", help="ID namespace prefix"),
    title: str = typer.Option("社内規定集（RAG用データ・拡張詳細版）", "--title", help="Document title"),
    batch_size: int = typer.Option(128, "--batch-size", help="Embedding API batch size"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip embeddings API calls"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging")
):
    """
    RAG Script: Process company policies into chunks and embeddings.
    """
    logger = setup_logging(verbose)
    start_time = time.time()
    
    # Validate inputs
    if not input.exists():
        typer.echo(f"Error: Input file {input} does not exist", err=True)
        raise typer.Exit(1)
    
    if model not in MODEL_DIMENSIONS:
        typer.echo(f"Error: Unsupported model {model}", err=True)
        raise typer.Exit(1)
    
    if not dry_run and not os.getenv("OPENAI_API_KEY"):
        typer.echo("Error: OPENAI_API_KEY environment variable is required", err=True)
        raise typer.Exit(1)
    
    logger.info(f"Starting RAG processing: {input} -> {output}")
    logger.info(f"Model: {model}, Target tokens: {target_tokens}, Batch size: {batch_size}")
    
    try:
        # 1. Load and normalize text
        with open(input, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        normalized_text = normalize_text(raw_text)
        doc_title, content_text = extract_title(normalized_text)
        
        # Use provided title or extracted title
        final_title = title if title != "社内規定集（RAG用データ・拡張詳細版）" else doc_title
        
        logger.info(f"Document title: {final_title}")
        
        # 2. Parse sections
        sections = parse_sections(content_text)
        logger.info(f"Found {len(sections)} sections")
        
        if verbose:
            for section in sections:
                logger.info(f"  {section.emoji} {section.title} (page {section.page})")
        
        # 3. Chunk sections
        all_chunks = []
        for section in sections:
            section_chunks = chunk_section(section, target_tokens, overlap_tokens, min_chars, max_chars)
            all_chunks.extend(section_chunks)
            logger.info(f"Section '{section.title}': {len(section_chunks)} chunks")
        
        logger.info(f"Total chunks: {len(all_chunks)}")
        
        if not all_chunks:
            typer.echo("Error: No chunks generated", err=True)
            raise typer.Exit(1)
        
        # 4. Generate embeddings
        embeddings, retry_count = embed_chunks(all_chunks, model, batch_size, dry_run, logger)
        
        # 5. Create records
        records = []
        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            record = chunk_to_record(chunk, embedding, namespace, final_title, i)
            records.append(record)
        
        # 6. Save outputs
        save_jsonl(records, output, logger)
        
        # 7. Generate and save report
        processing_time = time.time() - start_time
        processing_report = generate_report(all_chunks, embeddings, retry_count, 
                                          processing_time, model, str(input), str(output))
        save_report(processing_report, report, logger)
        
        # Success summary
        logger.info(f"Processing completed in {processing_time:.2f}s")
        logger.info(f"Generated {len(records)} records")
        logger.info(f"Average token count: {processing_report.avg_token_count:.1f}")
        logger.info(f"Failed embeddings: {processing_report.failed_chunks}")
        
        if dry_run:
            typer.echo(" Dry run completed - no embeddings generated")
        else:
            typer.echo(f" Processing completed: {len(records)} records saved")
    
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()