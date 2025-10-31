"""PDF parsing and text extraction utilities."""

import re
from typing import Optional, Dict
from datetime import datetime
import sys
import os


def extract_metadata_from_pdf_stream(pdf_stream) -> Dict[str, str]:
    """Extract metadata from a PDF file stream.
    
    Args:
        pdf_stream: PDF file stream (from Streamlit file uploader)
        
    Returns:
        Dictionary containing PDF metadata fields
    """
    try:
        # Try PyMuPDF (fitz) first - best metadata support
        import fitz  # pymupdf
    except ImportError:
        try:
            # Fallback to pypdf
            from pypdf import PdfReader
            return _extract_metadata_pypdf(pdf_stream, PdfReader)
        except ImportError:
            try:
                # Last resort: PyPDF2
                import PyPDF2
                return _extract_metadata_pypdf(pdf_stream, PyPDF2.PdfReader)
            except ImportError:
                raise ImportError(
                    "No PDF library found. Install one with: pip install pymupdf or pip install pypdf"
                )
    
    # Use PyMuPDF for metadata extraction
    return _extract_metadata_pymupdf(pdf_stream, fitz)


def _extract_metadata_pymupdf(pdf_stream, fitz_module) -> Dict[str, str]:
    """Extract metadata using PyMuPDF.
    
    Args:
        pdf_stream: PDF file stream
        fitz_module: fitz module from pymupdf
        
    Returns:
        Dictionary containing PDF metadata
    """
    pdf_stream.seek(0)
    doc = fitz_module.open(stream=pdf_stream.read(), filetype="pdf")
    
    metadata_dict = {}
    
    # Extract metadata
    meta = doc.metadata
    for key, value in meta.items():
        if value and value.strip():
            # Clean up the key name
            clean_key = key.lower().replace(' ', '_')
            metadata_dict[clean_key] = value
    
    # Map PyMuPDF metadata keys to standard names
    key_mapping = {
        'title': 'Title',
        'author': 'Author',
        'subject': 'Subject',
        'keywords': 'Keywords',
        'creator': 'Creator',
        'producer': 'Producer',
        'creationdate': 'CreationDate',
        'moddate': 'ModDate',
    }
    
    # Rename keys to match standard
    renamed_dict = {}
    for old_key, value in metadata_dict.items():
        new_key = key_mapping.get(old_key, old_key.title())
        renamed_dict[new_key] = value
    
    doc.close()
    return renamed_dict


def _extract_metadata_pypdf(pdf_stream, PdfReader) -> Dict[str, str]:
    """Extract metadata using pypdf.
    
    Args:
        pdf_stream: PDF file stream
        PdfReader: PdfReader class from pypdf
        
    Returns:
        Dictionary containing PDF metadata
    """
    pdf_stream.seek(0)
    pdf_reader = PdfReader(pdf_stream)
    
    metadata_dict = {}
    
    # Extract standard PDF metadata
    if hasattr(pdf_reader, 'metadata') and pdf_reader.metadata:
        for key, value in pdf_reader.metadata.items():
            # Remove leading '/' from keys
            clean_key = key.lstrip('/')
            # Decode if byte string
            if isinstance(value, bytes):
                try:
                    value = value.decode('utf-8', errors='ignore')
                except:
                    pass
            metadata_dict[clean_key] = value
    
    # Also try to get XMP metadata if available
    if hasattr(pdf_reader, 'xmp_metadata') and pdf_reader.xmp_metadata:
        xmp = pdf_reader.xmp_metadata
        if hasattr(xmp, 'dc_creator'):
            metadata_dict['Creator'] = ', '.join(xmp.dc_creator)
        if hasattr(xmp, 'dc_title'):
            metadata_dict['Title'] = ', '.join(xmp.dc_title)
        if hasattr(xmp, 'dc_subject'):
            metadata_dict['Subject'] = ', '.join(xmp.dc_subject)
        if hasattr(xmp, 'pdf_keywords'):
            metadata_dict['Keywords'] = xmp.pdf_keywords
        if hasattr(xmp, 'xmp_create_date'):
            metadata_dict['XMP_CreateDate'] = str(xmp.xmp_create_date)
        if hasattr(xmp, 'xmp_modify_date'):
            metadata_dict['XMP_ModifyDate'] = str(xmp.xmp_modify_date)
    
    # Parse dates into readable format
    if 'CreationDate' in metadata_dict:
        try:
            parsed_date = parse_pdf_date(metadata_dict['CreationDate'])
            if parsed_date:
                metadata_dict['CreationDate_parsed'] = parsed_date.strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass
    
    if 'ModDate' in metadata_dict:
        try:
            parsed_date = parse_pdf_date(metadata_dict['ModDate'])
            if parsed_date:
                metadata_dict['ModDate_parsed'] = parsed_date.strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass
    
    return metadata_dict


def extract_text_from_pdf_stream(pdf_stream) -> str:
    # 🔍 Debug: show Python environment info
    print("🔍 Python executable:", sys.executable)
    print("🔍 sys.path:", sys.path)
    print("🔍 Current working directory:", os.getcwd())

    """Extract text from a PDF file stream using the best available library.
    
    Args:
        pdf_stream: A binary PDF file-like object (e.g. from Streamlit uploader)
    
    Returns:
        str: Extracted text from all PDF pages, separated by newlines
    """
    try:
        # Try PyMuPDF (fitz) first - fastest and best text extraction
        import fitz  # pymupdf
        return _extract_text_pymupdf(pdf_stream, fitz)
    except ImportError:
        try:
            # Fallback to pypdf
            from pypdf import PdfReader
            return _extract_text_pypdf(pdf_stream, PdfReader)
        except ImportError:
            try:
                # Last resort: PyPDF2
                import PyPDF2
                return _extract_text_pypdf(pdf_stream, PyPDF2.PdfReader)
            except ImportError:
                raise ImportError(
                    "No PDF library found. Install one with: pip install pymupdf or pip install pypdf"
                )


def _extract_text_pymupdf(pdf_stream, fitz_module) -> str:
    """Extract text using PyMuPDF.
    
    Args:
        pdf_stream: PDF file stream
        fitz_module: fitz module from pymupdf
        
    Returns:
        Extracted text
    """
    pdf_stream.seek(0)
    doc = fitz_module.open(stream=pdf_stream.read(), filetype="pdf")
    
    full_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            full_text.append(text.strip())
    
    doc.close()
    return "\n\n".join(full_text)


def _extract_text_pypdf(pdf_stream, PdfReader) -> str:
    """Extract text using pypdf.
    
    Args:
        pdf_stream: PDF file stream
        PdfReader: PdfReader class from pypdf
        
    Returns:
        Extracted text
    """
    pdf_stream.seek(0)
    reader = PdfReader(pdf_stream)
    full_text = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():  # Only add non-empty text
            full_text.append(text.strip())

    return "\n\n".join(full_text)


def parse_pdf_date(pdf_date_str: str) -> Optional[datetime]:
    """Parse PDF date string to datetime object.
    
    Args:
        pdf_date_str: PDF date string in format D:YYYYMMDDHHMMSSOHH'mm'
        
    Returns:
        datetime object or None if parsing fails
    """
    if not pdf_date_str:
        return None
    
    try:
        # Regular expression to match the PDF date format
        pattern = r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})([+-Zz])?(\d{2})?'?(\d{2})?'?"
        match = re.match(pattern, pdf_date_str)
        
        if not match:
            return None
        
        # Extract the date and time components
        year, month, day, hour, minute, second, offset_sign, offset_hour, offset_minute = match.groups()
        dt = datetime(
            int(year), int(month), int(day),
            int(hour), int(minute), int(second)
        )
        
        # Handle the timezone offset
        if offset_sign and offset_sign != 'Z':
            if offset_hour and offset_minute:
                offset = int(offset_hour) * 60 + int(offset_minute)
                if offset_sign == '-':
                    offset = -offset
                from datetime import timedelta
                dt -= timedelta(minutes=offset)
        
        return dt
    except:
        return None


def clean_pdf_text(text: str) -> str:
    """Clean and filter extracted PDF text.
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        Cleaned and filtered text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers and footers (common patterns)
    text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
    
    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove special characters that might interfere with text processing
    # Keep only printable characters, newlines, and common punctuation
    text = re.sub(r'[^\x20-\x7E\n\t\u00A0-\uFFFF]', '', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def filter_text_content(text: str, min_length: int = 10) -> str:
    """Filter text content to remove noise and ensure quality.
    
    Args:
        text: Input text to filter
        min_length: Minimum length for a valid paragraph/section
        
    Returns:
        Filtered text
    """
    if not text:
        return ""
    
    # Split into lines and filter
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip very short lines that are likely headers or page artifacts
        if len(line) < min_length:
            continue
        
        # Skip lines that are just numbers or symbols
        if re.match(r'^[\d\s\-\.,;:]+$', line):
            continue
        
        filtered_lines.append(line)
    
    # Join back into paragraphs
    filtered_text = '\n\n'.join(filtered_lines)
    
    # Remove excessive blank lines between paragraphs
    filtered_text = re.sub(r'\n{3,}', '\n\n', filtered_text)
    
    return filtered_text.strip()


def process_pdf_content(pdf_stream, return_metadata=False):
    """Complete PDF processing pipeline: extract, clean, and filter.
    
    Args:
        pdf_stream: PDF file stream (from Streamlit file uploader)
        return_metadata: If True, returns (text, metadata), otherwise just text
        
    Returns:
        Processed text or tuple of (text, metadata) if return_metadata=True
    """
    # Extract raw text
    raw_text = extract_text_from_pdf_stream(pdf_stream)
    
    # Clean the text
    cleaned_text = clean_pdf_text(raw_text)
    
    # Filter for quality
    filtered_text = filter_text_content(cleaned_text)
    
    if return_metadata:
        # Also extract metadata
        pdf_stream.seek(0)  # Reset stream
        metadata = extract_metadata_from_pdf_stream(pdf_stream)
        return filtered_text, metadata
    
    return filtered_text
