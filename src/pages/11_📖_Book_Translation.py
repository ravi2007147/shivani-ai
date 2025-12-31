"""Book Translation Page - Translate books page by page.

This page allows you to:
- View book pages one at a time
- Enter translations for each page
- Navigate between pages
- Save translations automatically
"""

import streamlit as st
import sys
import logging
from pathlib import Path
from typing import Optional, List
import os
import requests
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.books_db import BooksDBManager
from src.utils.pdf_parser import extract_text_from_pdf_stream
from src.utils import start_api_server, is_api_server_running
from src.utils import start_api_server, is_api_server_running, get_api_server_url
from langchain_ollama import OllamaLLM
from src.config import DEFAULT_OLLAMA_BASE_URL
from langchain_ollama import OllamaLLM
from src.config import DEFAULT_OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Book Translation",
    page_icon="📖",
    layout="wide"
)


def extract_page_text_accurate(pdf_path: Path, page_number: int, cached_doc=None) -> Optional[str]:
    """Extract text from a specific page of a PDF with high accuracy.
    Preserves spacing, newlines, and tabs for proper formatting.
    Uses PyMuPDF's advanced text extraction methods for better accuracy.
    
    Args:
        pdf_path: Path to the PDF file
        page_number: Page number (0-indexed)
        cached_doc: Optional cached PDF document
        
    Returns:
        Extracted text from the page with preserved formatting, or None if error/no text
    """
    try:
        import fitz  # pymupdf
        
        # Use cached document if provided, otherwise open new one
        if cached_doc is not None:
            doc = cached_doc
            should_close = False
        else:
            doc = fitz.open(pdf_path)
            should_close = True
        
        if page_number >= len(doc):
            if should_close:
                doc.close()
            return None
        
        page = doc[page_number]
        
        # Method 1: Use "dict" format to preserve layout with spacing and newlines
        # This method preserves the exact layout including indentation
        try:
            text_dict = page.get_text("dict")
            if text_dict and "blocks" in text_dict:
                result_lines = []
                prev_y = None
                
                for block in text_dict["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            if "spans" in line:
                                line_text_parts = []
                                prev_x = None
                                
                                for span in line["spans"]:
                                    span_text = span.get("text", "")
                                    if not span_text:
                                        continue
                                    
                                    # Get position information
                                    bbox = span.get("bbox", [])
                                    if len(bbox) >= 4:
                                        x0 = bbox[0]
                                        
                                        # Detect indentation (tabs) based on x position
                                        if prev_x is not None and x0 > prev_x + 10:  # Significant x difference
                                            # Calculate approximate tabs (assuming ~50px per tab)
                                            tabs_needed = int((x0 - prev_x) / 50)
                                            if tabs_needed > 0:
                                                line_text_parts.append("\t" * min(tabs_needed, 4))  # Max 4 tabs
                                        
                                        prev_x = x0 + (bbox[2] - bbox[0])  # End of span
                                    
                                    line_text_parts.append(span_text)
                                
                                if line_text_parts:
                                    line_text = "".join(line_text_parts)
                                    if line_text.strip():
                                        result_lines.append(line_text)
                                        
                                        # Preserve vertical spacing
                                        bbox = line.get("bbox", [])
                                        if len(bbox) >= 4:
                                            current_y = bbox[1]
                                            if prev_y is not None:
                                                # If significant vertical gap, add extra newline
                                                y_gap = current_y - prev_y
                                                if y_gap > 20:  # Significant gap (likely paragraph break)
                                                    result_lines.append("")
                                            prev_y = current_y
                
                if result_lines:
                    result = "\n".join(result_lines)
                    if result.strip():
                        if should_close:
                            doc.close()
                        return result
        except Exception as e:
            logger.debug(f"Dict extraction method failed: {e}")
        
        # Method 2: Use "text" format - preserves newlines
        text = page.get_text("text")
        if text and text.strip():
            # Clean up but preserve newlines and spacing
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                cleaned = line.rstrip()  # Remove trailing spaces but keep leading
                if cleaned or cleaned_lines:  # Keep empty lines between paragraphs
                    cleaned_lines.append(cleaned)
            result = "\n".join(cleaned_lines)
            if result.strip():
                if should_close:
                    doc.close()
                return result
        
        # Method 3: Use blocks extraction - preserves paragraph structure
        blocks = page.get_text("blocks")
        if blocks:
            text_lines = []
            for block in blocks:
                if len(block) > 4:
                    block_text = block[4]
                    # Preserve the text as-is including spaces
                    if block_text.strip():
                        # Add newline after each block (paragraph)
                        text_lines.append(block_text.rstrip())
                        text_lines.append("")  # Empty line between blocks
            
            if text_lines:
                result = "\n".join(text_lines)
                if result.strip():
                    if should_close:
                        doc.close()
                    return result
        
        # Method 4: Use words extraction to reconstruct with spacing
        try:
            words = page.get_text("words")
            if words:
                result_text = []
                prev_word = None
                
                for word_info in words:
                    if len(word_info) < 5:
                        continue
                    
                    word_text = word_info[4]
                    x0, y0, x1, y1 = word_info[0], word_info[1], word_info[2], word_info[3]
                    
                    if prev_word:
                        prev_x1, prev_y0 = prev_word[2], prev_word[1]
                        
                        # Check if new line
                        if abs(y0 - prev_y0) > 5:  # Different line
                            result_text.append("\n")
                            # Check for indentation
                            if x0 > 50:  # Indented
                                tabs = int(x0 / 50)
                                result_text.append("\t" * min(tabs, 4))
                        # Check if space needed between words
                        elif x0 - prev_x1 > 3:  # Space between words
                            result_text.append(" ")
                    
                    result_text.append(word_text)
                    prev_word = word_info
                
                if result_text:
                    result = "".join(result_text)
                    if result.strip():
                        if should_close:
                            doc.close()
                        return result
        except Exception as e:
            logger.debug(f"Words extraction method failed: {e}")
        
        # Method 5: Standard text extraction (fallback)
        text = page.get_text()
        if text and text.strip():
            if should_close:
                doc.close()
            return text
        
        if should_close:
            doc.close()
        return None  # No text found
        
    except ImportError:
        try:
            # Fallback to pypdf
            from pypdf import PdfReader
            
            reader = PdfReader(pdf_path)
            if page_number >= len(reader.pages):
                return None
            
            page = reader.pages[page_number]
            text = page.extract_text()
            return text if text and text.strip() else None
            
        except ImportError:
            logger.error("No PDF library available")
            return None
    except Exception as e:
        logger.error(f"Error extracting page text: {e}")
        return None


def extract_page_text(pdf_path: Path, page_number: int, cached_doc=None) -> Optional[str]:
    """Extract text from a specific page of a PDF (legacy function for compatibility).
    
    Args:
        pdf_path: Path to the PDF file
        page_number: Page number (0-indexed)
        cached_doc: Optional cached PDF document
        
    Returns:
        Extracted text from the page, or None if error/no text
    """
    return extract_page_text_accurate(pdf_path, page_number, cached_doc)


def render_page_as_image(pdf_path: Path, page_number: int, zoom: float = 2.0, cached_doc=None) -> Optional[bytes]:
    """Render a PDF page as an image using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        page_number: Page number (0-indexed)
        zoom: Zoom factor for rendering (higher = better quality, default 2.0 for good readability)
        cached_doc: Optional cached PDF document to reuse
        
    Returns:
        Image bytes (PNG format) or None if error
    """
    try:
        import fitz  # pymupdf
        
        # Use cached document if provided, otherwise open new one
        if cached_doc is not None:
            doc = cached_doc
            should_close = False
        else:
            doc = fitz.open(pdf_path)
            should_close = True
        
        if page_number >= len(doc):
            if should_close:
                doc.close()
            return None
        
        page = doc[page_number]
        
        # Render page to image with high quality
        # Use zoom factor for better readability (2.0 = 200% zoom)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
        
        # Clean up pixmap
        pix = None
        
        if should_close:
            doc.close()
        return img_bytes
        
    except ImportError:
        logger.warning("PyMuPDF not available for image rendering")
        return None
    except Exception as e:
        logger.error(f"Error rendering page as image: {e}")
        return None


def get_total_pages(pdf_path: Path) -> int:
    """Get total number of pages in a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Total number of pages
    """
    try:
        import fitz  # pymupdf
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        return page_count
    except ImportError:
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            return len(reader.pages)
        except ImportError:
            return 0
    except Exception as e:
        logger.error(f"Error getting page count: {e}")
        return 0


def format_text_for_display(text: str) -> str:
    """Format text for better readability in Streamlit.
    
    Args:
        text: Raw text from PDF
        
    Returns:
        Formatted text
    """
    if not text:
        return ""
    
    # Clean up excessive whitespace
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        cleaned = line.strip()
        if cleaned:
            cleaned_lines.append(cleaned)
        elif cleaned_lines and cleaned_lines[-1]:  # Add single blank line between paragraphs
            cleaned_lines.append("")
    
    return '\n'.join(cleaned_lines)


# Initialize books database manager
books_db = BooksDBManager()

# Ensure API server is running (needed for copy functionality)
if not is_api_server_running():
    try:
        if start_api_server():
            logger.info("API server started for PDF text extraction")
        else:
            logger.warning("Could not start API server")
    except Exception as e:
        logger.error(f"Error starting API server: {e}")

# Get book_id from query parameters or session state
book_id = None

# Try query parameters first
if 'book_id' in st.query_params:
    try:
        book_id = int(st.query_params['book_id'])
    except (ValueError, KeyError):
        pass

# Try session state
if book_id is None and 'selected_book_id' in st.session_state:
    book_id = st.session_state.selected_book_id

if book_id is None:
    st.error("❌ No book selected. Please go back to Books Management and select a book.")
    if st.button("🏠 Go to Books Management"):
        st.switch_page("pages/10_📖_Books_Management.py")
    st.stop()

# Get book information
book = books_db.get_book(book_id)
if not book:
    st.error("❌ Book not found.")
    st.stop()

# Initialize session state for current page with book-specific key
# This ensures each book has its own page position
current_page_key = f'current_page_{book_id}'
current_book_id_key = f'current_book_id_{book_id}'

# Only reset if book_id changed, otherwise preserve current_page across reruns
if st.session_state.get('current_book_id') != book_id:
    # Book changed, reset to first page for this book
    st.session_state[current_page_key] = 0
    st.session_state.current_book_id = book_id
    st.session_state[current_book_id_key] = book_id
elif current_page_key not in st.session_state:
    # First time for this book, initialize to 0
    st.session_state[current_page_key] = 0
    st.session_state.current_book_id = book_id
    st.session_state[current_book_id_key] = book_id

# Get current page from session state (preserved across reruns)
# Always use the book-specific key to ensure it persists
current_page = st.session_state.get(current_page_key, 0)

# Get PDF file path
pdf_path = Path(book['file_path'])
if not pdf_path.exists():
    st.error(f"❌ PDF file not found: {pdf_path}")
    st.stop()

# Get total pages
total_pages = get_total_pages(pdf_path)
if total_pages == 0:
    st.error("❌ Could not read PDF file or PDF has no pages.")
    st.stop()

# Ensure current_page is within bounds and update session state
if current_page >= total_pages:
    current_page = total_pages - 1
if current_page < 0:
    current_page = 0

# Always update session state to preserve current page (using book-specific key)
# This ensures the page position persists even when textarea causes reruns
st.session_state[current_page_key] = current_page
st.session_state.current_book_id = book_id
st.session_state[current_book_id_key] = book_id

# Cache PDF document in session state to avoid reopening
pdf_cache_key = f"pdf_doc_{book_id}"
if pdf_cache_key not in st.session_state:
    try:
        import fitz  # pymupdf
        st.session_state[pdf_cache_key] = fitz.open(pdf_path)
        st.session_state[f"{pdf_cache_key}_path"] = str(pdf_path)
    except Exception as e:
        logger.error(f"Error caching PDF: {e}")
        st.session_state[pdf_cache_key] = None
else:
    # Verify cached document is still valid
    cached_doc = st.session_state.get(pdf_cache_key)
    if cached_doc is not None:
        try:
            # Check if document is still open and valid
            _ = len(cached_doc)
        except:
            # Document was closed or corrupted, reopen it
            try:
                import fitz  # pymupdf
                st.session_state[pdf_cache_key] = fitz.open(pdf_path)
                st.session_state[f"{pdf_cache_key}_path"] = str(pdf_path)
            except Exception as e:
                logger.error(f"Error reopening cached PDF: {e}")
                st.session_state[pdf_cache_key] = None

# Page header
st.title(f"📖 {book.get('title') or book.get('filename', 'Book Translation')}")
if book.get('author'):
    st.caption(f"by {book['author']}")

# Auto-translation helper functions
def translate_text_with_ollama(llm: OllamaLLM, text: str) -> str:
    """Translate English text to Hindi using Ollama LLM."""
    translation_prompt = f"""You are a professional translator with native-level proficiency in both English and Hindi. Your expertise includes understanding cultural nuances, idiomatic expressions, and context-dependent meanings.

TASK: Translate the following English text into natural, fluent Hindi that reads as if it were originally written in Hindi.

TRANSLATION GUIDELINES:
1. **Context Understanding**: Analyze the full context, not individual words. Understand the intended meaning, tone, and purpose.
2. **Natural Flow**: The translation should flow naturally in Hindi, using appropriate sentence structures and word order.
3. **Cultural Adaptation**: Adapt cultural references, idioms, and expressions to be meaningful in Hindi context.
4. **Vocabulary Selection**: Choose the most appropriate Hindi words that convey the exact meaning and register (formal/informal).
5. **Grammar & Syntax**: Use correct Hindi grammar, including proper use of matras (diacritics), verb conjugations, and case markers.
6. **Technical Terms**: For technical terms, use standard Hindi translations when available, or transliterate appropriately.
7. **Proper Nouns**: Keep names, places, and brand names as-is or use common Hindi transliterations.
8. **Tone Preservation**: Maintain the original tone (formal, casual, technical, narrative, etc.).

ORIGINAL ENGLISH TEXT:
{text}

TRANSLATION REQUIREMENTS:
- Output ONLY the Hindi translation
- No explanations, notes, or additional text
- Preserve paragraph breaks and formatting
- Use proper Devanagari script

Hindi Translation:"""
    
    try:
        translation = llm.invoke(translation_prompt).strip()
        prefixes_to_remove = [
            "Hindi translation:", "Translation:", "Here is the translation:",
            "The Hindi translation is:", "हिंदी अनुवाद:", "अनुवाद:",
            "Hindi Translation:", "TRANSLATION:"
        ]
        for prefix in prefixes_to_remove:
            if translation.lower().startswith(prefix.lower()):
                translation = translation[len(prefix):].strip()
        return translation
    except Exception as e:
        logger.error(f"Translation error: {e}", exc_info=True)
        raise


def translate_with_refinement(llm: OllamaLLM, text: str) -> str:
    """Translate with two-step process: initial translation + refinement."""
    initial_prompt = f"""Translate the following English text to Hindi. Focus on accuracy and meaning.

English text:
{text}

Provide the Hindi translation:"""
    
    try:
        initial_translation = llm.invoke(initial_prompt).strip()
        prefixes_to_remove = ["Hindi translation:", "Translation:", "Here is the translation:", "हिंदी अनुवाद:", "अनुवाद:"]
        for prefix in prefixes_to_remove:
            if initial_translation.lower().startswith(prefix.lower()):
                initial_translation = initial_translation[len(prefix):].strip()
        
        refinement_prompt = f"""You are refining a Hindi translation to make it more natural, fluent, and accurate.

ORIGINAL ENGLISH TEXT:
{text}

CURRENT HINDI TRANSLATION:
{initial_translation}

TASK: Refine this translation to:
1. Make it sound more natural and fluent in Hindi
2. Improve word choice and expressions
3. Ensure proper grammar and sentence structure
4. Enhance readability and flow
5. Fix any awkward phrasings or literal translations

Output ONLY the refined Hindi translation, no explanations:"""
        
        refined_translation = llm.invoke(refinement_prompt).strip()
        for prefix in prefixes_to_remove:
            if refined_translation.lower().startswith(prefix.lower()):
                refined_translation = refined_translation[len(prefix):].strip()
        return refined_translation
    except Exception as e:
        logger.error(f"Refined translation error: {e}", exc_info=True)
        return translate_text_with_ollama(llm, text)


# Generate PDF, PWA Package, Dashboard, and Auto Translate buttons at the top
col_pdf1, col_pdf2, col_pdf3, col_pdf4, col_pdf5 = st.columns([1.2, 1, 1, 1, 1.2])
with col_pdf2:
    # Create Generate PDF button with JavaScript
    generate_pdf_link_id = f"generate_pdf_link_{book_id}"
    generate_pdf_func_name = f"generateTranslatedPDF_{book_id}"
    
    generate_pdf_html = f"""
    <script>
    (function() {{
        function {generate_pdf_func_name}(e) {{
            if (e) {{
                e.preventDefault();
                e.stopPropagation();
                e.stopImmediatePropagation();
            }}
            
            const pdfLink = document.getElementById('{generate_pdf_link_id}');
            if (!pdfLink) {{
                console.error('Generate PDF link not found');
                return false;
            }}
            
            console.log('Generating PDF for book_id={book_id}');
            
            // Show loading state
            const originalText = pdfLink.textContent;
            pdfLink.textContent = '⏳ Generating PDF...';
            pdfLink.style.pointerEvents = 'none';
            pdfLink.style.color = '#ff9800';
            
            // Call API endpoint to generate PDF
            fetch('http://127.0.0.1:8000/api/pdf/generate-translated-pdf', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify({{
                    book_id: {book_id}
                }})
            }})
            .then(response => {{
                console.log('PDF generation response status:', response.status);
                if (!response.ok) {{
                    throw new Error('HTTP error! status: ' + response.status);
                }}
                
                // Get filename from Content-Disposition header or use default
                const contentDisposition = response.headers.get('Content-Disposition');
                let filename = 'translated_book.pdf';
                if (contentDisposition) {{
                    const filenameMatch = contentDisposition.match(/filename="?(.+?)"?$/);
                    if (filenameMatch) {{
                        filename = filenameMatch[1];
                    }}
                }}
                
                // Convert response to blob
                return response.blob().then(blob => ({{
                    blob: blob,
                    filename: filename
                }}));
            }})
            .then(({{blob, filename}}) => {{
                console.log('PDF generated successfully, size:', blob.size);
                
                // Create download link
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                // Show success message
                pdfLink.textContent = '✅ PDF Downloaded!';
                pdfLink.style.color = '#28a745';
                setTimeout(function() {{
                    pdfLink.textContent = originalText;
                    pdfLink.style.color = '#1f77b4';
                    pdfLink.style.pointerEvents = 'auto';
                }}, 3000);
            }})
            .catch(error => {{
                console.error('PDF generation error:', error);
                alert('Failed to generate PDF: ' + error.message + '. Please check if API server is running at http://127.0.0.1:8000');
                pdfLink.textContent = originalText;
                pdfLink.style.color = '#1f77b4';
                pdfLink.style.pointerEvents = 'auto';
            }});
            
            return false;
        }}
        
        // Make function globally accessible
        window.{generate_pdf_func_name} = {generate_pdf_func_name};
    }})();
    </script>
    <a href="javascript:void(0)" 
       id="{generate_pdf_link_id}" 
       onclick="return {generate_pdf_func_name}(event);"
       style="text-decoration: none; color: #1f77b4; cursor: pointer; font-size: 1em; display: inline-block; font-weight: bold; padding: 8px 16px; border: 2px solid #1f77b4; border-radius: 4px; background-color: #f0f8ff;">
       📄 Generate PDF
    </a>
    """
    from streamlit.components.v1 import html
    html(generate_pdf_html, height=50)

with col_pdf3:
    # Create Generate PWA Package button
    generate_pwa_link_id = f"generate_pwa_link_{book_id}"
    generate_pwa_func_name = f"generatePWAPackage_{book_id}"
    
    generate_pwa_html = f"""
    <script>
    (function() {{
        function {generate_pwa_func_name}(e) {{
            if (e) {{
                e.preventDefault();
                e.stopPropagation();
                e.stopImmediatePropagation();
            }}
            
            const pwaLink = document.getElementById('{generate_pwa_link_id}');
            if (!pwaLink) {{
                console.error('Generate PWA link not found');
                return false;
            }}
            
            console.log('Generating PWA package for book_id={book_id}');
            
            // Show loading state
            const originalText = pwaLink.textContent;
            pwaLink.textContent = '⏳ Generating...';
            pwaLink.style.pointerEvents = 'none';
            pwaLink.style.color = '#ff9800';
            
            // Call API endpoint to generate PWA package
            fetch('http://127.0.0.1:8000/api/pwa/generate-package', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify({{
                    book_id: {book_id},
                    base_url: 'https://blog.priorcoder.com/books'
                }})
            }})
            .then(response => {{
                console.log('PWA generation response status:', response.status);
                if (!response.ok) {{
                    throw new Error('HTTP error! status: ' + response.status);
                }}
                
                // Get filename from Content-Disposition header
                const contentDisposition = response.headers.get('Content-Disposition');
                let filename = 'book_pwa.zip';
                if (contentDisposition) {{
                    const filenameMatch = contentDisposition.match(/filename="?(.+?)"?$/);
                    if (filenameMatch) {{
                        filename = filenameMatch[1];
                    }}
                }}
                
                // Convert response to blob
                return response.blob().then(blob => ({{
                    blob: blob,
                    filename: filename
                }}));
            }})
            .then(({{blob, filename}}) => {{
                console.log('PWA package generated successfully, size:', blob.size);
                
                // Create download link
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                // Show success message
                pwaLink.textContent = '✅ Downloaded!';
                pwaLink.style.color = '#28a745';
                setTimeout(function() {{
                    pwaLink.textContent = originalText;
                    pwaLink.style.color = '#1f77b4';
                    pwaLink.style.pointerEvents = 'auto';
                }}, 3000);
            }})
            .catch(error => {{
                console.error('PWA generation error:', error);
                alert('Failed to generate PWA package: ' + error.message + '. Please check if API server is running at http://127.0.0.1:8000');
                pwaLink.textContent = originalText;
                pwaLink.style.color = '#1f77b4';
                pwaLink.style.pointerEvents = 'auto';
            }});
            
            return false;
        }}
        
        // Make function globally accessible
        window.{generate_pwa_func_name} = {generate_pwa_func_name};
    }})();
    </script>
    <a href="javascript:void(0)" 
       id="{generate_pwa_link_id}" 
       onclick="return {generate_pwa_func_name}(event);"
       style="text-decoration: none; color: #1f77b4; cursor: pointer; font-size: 1em; display: inline-block; font-weight: bold; padding: 8px 16px; border: 2px solid #1f77b4; border-radius: 4px; background-color: #f0f8ff;">
       📱 Generate PWA
    </a>
    """
    from streamlit.components.v1 import html
    html(generate_pwa_html, height=50)

# Add a separate row for Dashboard and Auto Translate buttons
col_dash1, col_dash2, col_dash3 = st.columns([3, 1, 1])
with col_dash2:
    # Create Generate Dashboard button
    generate_dashboard_link_id = f"generate_dashboard_link"
    generate_dashboard_func_name = f"generateDashboard"
    
    generate_dashboard_html = f"""
    <script>
    (function() {{
        function {generate_dashboard_func_name}(e) {{
            if (e) {{
                e.preventDefault();
                e.stopPropagation();
                e.stopImmediatePropagation();
            }}
            
            const dashboardLink = document.getElementById('{generate_dashboard_link_id}');
            if (!dashboardLink) {{
                console.error('Generate Dashboard link not found');
                return false;
            }}
            
            console.log('Generating Dashboard HTML');
            
            // Show loading state
            const originalText = dashboardLink.textContent;
            dashboardLink.textContent = '⏳ Generating...';
            dashboardLink.style.pointerEvents = 'none';
            dashboardLink.style.color = '#ff9800';
            
            // Call API endpoint to generate dashboard (returns ZIP file)
            fetch('http://127.0.0.1:8000/api/pwa/generate-dashboard?base_url=' + encodeURIComponent('https://blog.priorcoder.com/books'), {{
                method: 'GET',
                headers: {{
                    'Accept': 'application/zip',
                }}
            }})
            .then(response => {{
                console.log('Dashboard generation response status:', response.status);
                console.log('Content-Type:', response.headers.get('Content-Type'));
                if (!response.ok) {{
                    throw new Error('HTTP error! status: ' + response.status);
                }}
                return response.blob();
            }})
            .then(blob => {{
                console.log('Dashboard ZIP generated successfully, size:', blob.size, 'bytes');
                
                // Create download link for ZIP file
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'dashboard.zip';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                // Show success message
                dashboardLink.textContent = '✅ Downloaded!';
                dashboardLink.style.color = '#28a745';
                setTimeout(function() {{
                    dashboardLink.textContent = originalText;
                    dashboardLink.style.color = '#1f77b4';
                    dashboardLink.style.pointerEvents = 'auto';
                }}, 3000);
            }})
            .catch(error => {{
                console.error('Dashboard generation error:', error);
                alert('Failed to generate dashboard: ' + error.message + '. Please check if API server is running at http://127.0.0.1:8000');
                dashboardLink.textContent = originalText;
                dashboardLink.style.color = '#1f77b4';
                dashboardLink.style.pointerEvents = 'auto';
            }});
            
            return false;
        }}
        
        // Make function globally accessible
        window.{generate_dashboard_func_name} = {generate_dashboard_func_name};
    }})();
    </script>
    <a href="javascript:void(0)" 
       id="{generate_dashboard_link_id}" 
       onclick="return {generate_dashboard_func_name}(event);"
       style="text-decoration: none; color: #1f77b4; cursor: pointer; font-size: 0.9em; display: inline-block; font-weight: bold; padding: 8px 16px; border: 2px solid #1f77b4; border-radius: 4px; background-color: #f0f8ff;">
       📊 Get Dashboard
    </a>
    """
    from streamlit.components.v1 import html
    html(generate_dashboard_html, height=50)

# Auto Translate button (AJAX-based, no page refresh)
col_auto1, col_auto2 = st.columns([4, 1])
with col_auto2:
    auto_translate_link_id = f"auto_translate_link_{book_id}"
    auto_translate_func_name = f"startAutoTranslate_{book_id}"
    
    auto_translate_html = f"""
    <div>
    <script>
    (function() {{
        let isPolling = false;
        let pollInterval = null;
        let jobId = null;
        const bookId = {book_id};
        const totalPages = {total_pages};
        const API_BASE = 'http://127.0.0.1:8000';
        
        // Helper function to get element (works in iframe or main window)
        function getElement(id) {{
            let el = document.getElementById(id);
            if (el) return el;
            if (window.parent && window.parent !== window) {{
                try {{
                    el = window.parent.document.getElementById(id);
                    if (el) return el;
                }} catch (e) {{
                    console.warn('Cannot access parent document:', e);
                }}
            }}
            return null;
        }}
        
        // Function to close progress dialog
        function closeProgressDialog() {{
            const el = getElement('auto_translate_progress');
            if (el) {{
                el.style.display = 'none';
            }}
        }}
        
        // Function to resume a stuck translation job
        function resumeTranslationJob(jobIdParam) {{
            console.log('🔄 Resuming translation job:', jobIdParam);
            const resumeBtn = getElement('auto_translate_resume_btn');
            if (resumeBtn) {{
                resumeBtn.textContent = '⏳ Resuming...';
                resumeBtn.style.pointerEvents = 'none';
            }}
            
            fetch(API_BASE + '/api/pdf/resume-translation-job/' + jobIdParam, {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json'
                }}
            }})
            .then(response => {{
                if (!response.ok) {{
                    throw new Error('HTTP error! status: ' + response.status);
                }}
                return response.json();
            }})
            .then(data => {{
                console.log('📥 Resume response:', data);
                if (data.success) {{
                    console.log('✅ Job resumed successfully');
                    jobId = jobIdParam;
                    isPolling = true;
                    // Start polling immediately
                    pollJobProgress();
                }} else {{
                    console.error('❌ Failed to resume job:', data.message || data.detail);
                    if (resumeBtn) {{
                        resumeBtn.textContent = '❌ Failed';
                        resumeBtn.style.background = 'rgba(244,67,54,0.8)';
                        resumeBtn.style.pointerEvents = 'auto';
                    }}
                }}
            }})
            .catch(error => {{
                console.error('❌ Error resuming job:', error);
                if (resumeBtn) {{
                    resumeBtn.textContent = '❌ Error';
                    resumeBtn.style.background = 'rgba(244,67,54,0.8)';
                    resumeBtn.style.pointerEvents = 'auto';
                }}
            }});
        }}
        
        // Function to restart a translation job from the beginning
        function restartTranslationJob(jobIdParam) {{
            console.log('🔄 Restarting translation job:', jobIdParam);
            const restartBtn = getElement('auto_translate_restart_btn');
            if (restartBtn) {{
                restartBtn.textContent = '⏳ Restarting...';
                restartBtn.style.pointerEvents = 'none';
            }}
            
            fetch(API_BASE + '/api/pdf/restart-translation-job/' + jobIdParam, {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json'
                }}
            }})
            .then(response => {{
                if (!response.ok) {{
                    throw new Error('HTTP error! status: ' + response.status);
                }}
                return response.json();
            }})
            .then(data => {{
                console.log('📥 Restart response:', data);
                if (data.success) {{
                    console.log('✅ Job restarted successfully, new job ID:', data.new_job_id);
                    jobId = data.new_job_id;
                    isPolling = true;
                    // Refresh the page to show the new job
                    setTimeout(() => window.location.reload(), 1000);
                }} else {{
                    console.error('❌ Failed to restart job:', data.message || data.detail);
                    if (restartBtn) {{
                        restartBtn.textContent = '❌ Failed';
                        restartBtn.style.background = 'rgba(244,67,54,0.8)';
                        restartBtn.style.pointerEvents = 'auto';
                    }}
                }}
            }})
            .catch(error => {{
                console.error('❌ Error restarting job:', error);
                if (restartBtn) {{
                    restartBtn.textContent = '❌ Error';
                    restartBtn.style.background = 'rgba(244,67,54,0.8)';
                    restartBtn.style.pointerEvents = 'auto';
                }}
            }});
        }}
        
        // Make resume and restart functions globally accessible (both in current scope and window)
        window.resumeTranslationJob = resumeTranslationJob;
        window.restartTranslationJob = restartTranslationJob;
        // Also make it accessible in parent window if in iframe
        if (window.parent && window.parent !== window) {{
            try {{
                window.parent.resumeTranslationJob = resumeTranslationJob;
                window.parent.restartTranslationJob = restartTranslationJob;
            }} catch (e) {{
                console.warn('Cannot set functions in parent window:', e);
            }}
        }}
        
        // Create or get progress container
        function getOrCreateProgressContainer() {{
            let progressContainer = getElement('auto_translate_progress');
            if (!progressContainer) {{
                progressContainer = document.createElement('div');
                progressContainer.id = 'auto_translate_progress';
                progressContainer.style.cssText = 'position: fixed; top: 0; left: 50%; transform: translateX(-50%); width: 80%; max-width: 800px; margin: 20px 0; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; box-shadow: 0 6px 12px rgba(0,0,0,0.3); color: white; z-index: 9999; display: block;';
                
                // Try to insert in parent window if in iframe
                if (window.parent && window.parent !== window) {{
                    try {{
                        window.parent.document.body.appendChild(progressContainer);
                        console.log('✅ Progress bar created in parent window');
                    }} catch (e) {{
                        document.body.appendChild(progressContainer);
                        console.log('✅ Progress bar created in current window (fallback)');
                    }}
                }} else {{
                    document.body.appendChild(progressContainer);
                    console.log('✅ Progress bar created in current window');
                }}
            }} else {{
                // Ensure it's visible
                progressContainer.style.display = 'block';
            }}
            return progressContainer;
        }}
        
        // Update progress display
        function updateProgress(progress) {{
            const container = getOrCreateProgressContainer();
            // Ensure container is visible
            container.style.display = 'block';
            const completed = progress.completed_pages || 0;
            const failed = progress.failed_pages || 0;
            const total = progress.total_pages || totalPages;
            const percent = total > 0 ? Math.round((completed / total) * 100) : 0;
            const avgTime = progress.avg_time_per_page || 0;
            const estTime = progress.estimated_time_remaining || 0;
            
            let statusText = '⏳ Starting...';
            if (progress.status === 'running' || progress.status === 'pending') statusText = '🔄 Translation in progress...';
            else if (progress.status === 'stuck') statusText = '⚠️ Translation job is stuck (server restarted)';
            else if (progress.status === 'running' || progress.status === 'pending') statusText = '🔄 Translation in progress...';
            else if (progress.status === 'completed') statusText = '✅ Translation completed!';
            else if (progress.status === 'failed') statusText = '❌ Translation failed';
            
            let avgTimeText = '--';
            if (avgTime > 0) {{
                if (avgTime < 60) avgTimeText = avgTime.toFixed(1) + 's';
                else avgTimeText = (avgTime / 60).toFixed(1) + ' min';
            }}
            
            let estTimeText = '--';
            if (estTime > 0) {{
                if (estTime < 60) estTimeText = Math.ceil(estTime) + 's';
                else if (estTime < 3600) estTimeText = Math.ceil(estTime / 60) + ' min';
                else {{
                    const hours = Math.floor(estTime / 3600);
                    const minutes = Math.ceil((estTime % 3600) / 60);
                    estTimeText = hours + 'h ' + minutes + 'm';
                }}
            }}
            
            container.innerHTML = '<div style="margin-bottom: 20px;">' +
                '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">' +
                '<span style="font-size: 1.2em; font-weight: bold; color: white;">🤖 Auto Translation Progress</span>' +
                '<div style="display: flex; align-items: center; gap: 10px;">' +
                '<span id="auto_translate_status" style="font-size: 1em; background: rgba(255,255,255,0.3); padding: 8px 15px; border-radius: 20px; font-weight: bold; color: white;">' + completed + ' / ' + total + ' pages</span>' +
                (progress.status === 'stuck' ? 
                    '<button id="auto_translate_resume_btn" data-job-id="' + progress.job_id + '" style="background: rgba(76,175,80,0.8); border: none; color: white; padding: 8px 15px; border-radius: 20px; cursor: pointer; font-weight: bold; font-size: 0.9em; margin-right: 10px;">▶ Resume</button>' + 
                    '<button id="auto_translate_restart_btn" data-job-id="' + progress.job_id + '" style="background: rgba(255,152,0,0.8); border: none; color: white; padding: 8px 15px; border-radius: 20px; cursor: pointer; font-weight: bold; font-size: 0.9em; margin-right: 10px;">🔄 Restart</button>' : 
                    '') +
                ((progress.status === 'running' || progress.status === 'pending') ? 
                    '<button id="auto_translate_restart_btn" data-job-id="' + progress.job_id + '" style="background: rgba(255,152,0,0.8); border: none; color: white; padding: 8px 15px; border-radius: 20px; cursor: pointer; font-weight: bold; font-size: 0.9em; margin-right: 10px;">🔄 Restart</button>' : 
                    '') +
                (progress.status === 'completed' || progress.status === 'failed' ? 
                    '<button id="auto_translate_close_btn" onclick="closeProgressDialog()" style="background: rgba(255,255,255,0.3); border: none; color: white; padding: 8px 15px; border-radius: 20px; cursor: pointer; font-weight: bold; font-size: 0.9em;">✕ Close</button>' : 
                    '') +
                '</div>' +
                '</div>' +
                '<div style="width: 100%; background: rgba(255,255,255,0.3); border-radius: 10px; height: 40px; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);">' +
                '<div id="auto_translate_progress_bar" style="width: ' + percent + '%; background: linear-gradient(90deg, #4CAF50, #45a049); height: 100%; transition: width 0.8s ease; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 1em;">' +
                '<span id="auto_translate_percent">' + percent + '%</span>' +
                '</div>' +
                '</div>' +
                '</div>' +
                '<div id="auto_translate_stats" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; font-size: 1em; margin-top: 10px;">' +
                '<div style="background: rgba(76,175,80,0.2); padding: 12px 15px; border-radius: 10px; text-align: center;">' +
                '<div style="font-size: 0.85em; opacity: 0.8; margin-bottom: 5px;">✅ Translated:</div>' +
                '<div id="auto_translated_count" style="font-size: 1.8em; font-weight: bold; color: #4CAF50;">' + completed + '</div>' +
                '</div>' +
                '<div style="background: rgba(244,67,54,0.2); padding: 12px 15px; border-radius: 10px; text-align: center;">' +
                '<div style="font-size: 0.85em; opacity: 0.8; margin-bottom: 5px;">❌ Failed:</div>' +
                '<div id="auto_failed_count" style="font-size: 1.8em; font-weight: bold; color: #F44336;">' + failed + '</div>' +
                '</div>' +
                '<div style="background: rgba(33,150,243,0.2); padding: 12px 15px; border-radius: 10px; text-align: center;">' +
                '<div style="font-size: 0.85em; opacity: 0.8; margin-bottom: 5px;">📊 Total:</div>' +
                '<div id="auto_total_count" style="font-size: 1.8em; font-weight: bold; color: #2196F3;">' + total + '</div>' +
                '</div>' +
                '</div>' +
                '<div id="auto_translate_current" style="background: rgba(0,0,0,0.2); padding: 15px; border-radius: 10px; margin-top: 10px;">' +
                '<div style="font-size: 0.85em; opacity: 0.8; margin-bottom: 8px; color: rgba(255,255,255,0.9);">Current Status:</div>' +
                '<div id="auto_translate_current_text" style="font-size: 1.1em; font-weight: 500; color: white;">' + statusText + '</div>' +
                '</div>' +
                '<div id="auto_translate_timing" style="background: rgba(0,0,0,0.2); padding: 15px; border-radius: 10px; margin-top: 10px; display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">' +
                '<div>' +
                '<div style="font-size: 0.85em; opacity: 0.8; margin-bottom: 5px; color: rgba(255,255,255,0.9);">⏱️ Avg Time/Page:</div>' +
                '<div id="auto_translate_avg_time" style="font-size: 1.3em; font-weight: bold; color: #FFC107;">' + avgTimeText + '</div>' +
                '</div>' +
                '<div>' +
                '<div style="font-size: 0.85em; opacity: 0.8; margin-bottom: 5px; color: rgba(255,255,255,0.9);">⏳ Est. Time Remaining:</div>' +
                '<div id="auto_translate_est_time" style="font-size: 1.3em; font-weight: bold; color: #4CAF50;">' + estTimeText + '</div>' +
                '</div>' +
                '</div>';
            
            // Attach event listener to resume button if it exists
            setTimeout(function() {{
                const resumeBtn = getElement('auto_translate_resume_btn');
                if (resumeBtn && !resumeBtn._hasClickListener) {{
                    const jobIdToResume = resumeBtn.getAttribute('data-job-id');
                    resumeBtn.addEventListener('click', function(e) {{
                        e.preventDefault();
                        e.stopPropagation();
                        if (jobIdToResume) {{
                            resumeTranslationJob(parseInt(jobIdToResume));
                        }}
                        return false;
                    }});
                    resumeBtn._hasClickListener = true;
                }}
                
                // Attach event listener to restart button if it exists
                const restartBtn = getElement('auto_translate_restart_btn');
                if (restartBtn && !restartBtn._hasClickListener) {{
                    const jobIdToRestart = restartBtn.getAttribute('data-job-id');
                    restartBtn.addEventListener('click', function(e) {{
                        e.preventDefault();
                        e.stopPropagation();
                        if (jobIdToRestart && confirm('Are you sure you want to restart the translation? This will delete all existing translations and start from the beginning.')) {{
                            restartTranslationJob(parseInt(jobIdToRestart));
                        }}
                        return false;
                    }});
                    restartBtn._hasClickListener = true;
                }}
            }}, 100);
        }}
        
        // Poll for job progress
        function pollJobProgress() {{
            if (!jobId) {{
                // Try to get active job for this book
                fetch(API_BASE + '/api/pdf/translation-job-for-book/' + bookId)
                    .then(response => {{
                        if (response.status === 404) {{
                            stopPolling();
                            return null;
                        }}
                        return response.json();
                    }})
                    .then(data => {{
                        if (data && data.job_id) {{
                            jobId = data.job_id;
                            updateProgress(data);
                            isPolling = true;
                            if (data.status === 'running' || data.status === 'pending') {{
                                pollInterval = setTimeout(pollJobProgress, 1000);
                            }} else if (data.status === 'stuck') {{
                                // Don't poll for stuck jobs - wait for user to resume
                                stopPolling();
                            }} else {{
                                stopPolling();
                                if (data.status === 'completed') {{
                                    setTimeout(() => window.location.reload(), 3000);
                                }}
                            }}
                        }} else {{
                            stopPolling();
                        }}
                    }})
                    .catch(error => {{
                        console.error('Error polling job progress:', error);
                        // Continue polling even on error (retry after 1 second)
                        if (isPolling) {{
                            pollInterval = setTimeout(pollJobProgress, 1000);
                        }}
                    }});
            }} else {{
                // Poll specific job
                fetch(API_BASE + '/api/pdf/translation-job-progress/' + jobId)
                    .then(response => {{
                        if (!response.ok) {{
                            throw new Error('HTTP error! status: ' + response.status);
                        }}
                        return response.json();
                    }})
                    .then(data => {{
                        updateProgress(data);
                        if (data.status === 'running' || data.status === 'pending') {{
                            pollInterval = setTimeout(pollJobProgress, 1000);
                        }} else if (data.status === 'stuck') {{
                            // Don't poll for stuck jobs - wait for user to resume
                            stopPolling();
                        }} else {{
                            stopPolling();
                            if (data.status === 'completed') {{
                                setTimeout(() => window.location.reload(), 3000);
                            }}
                        }}
                    }})
                    .catch(error => {{
                        console.error('Error polling job progress:', error);
                        // Continue polling even on error (retry after 1 second)
                        if (isPolling) {{
                            pollInterval = setTimeout(pollJobProgress, 1000);
                        }}
                    }});
            }}
        }}
        
        // Stop polling
        function stopPolling() {{
            if (pollInterval) {{
                clearTimeout(pollInterval);
                pollInterval = null;
            }}
            isPolling = false;
        }}
        
        // Start translation job
        function {auto_translate_func_name}(e) {{
            console.log('🚀 Auto Translate button clicked!', e);
            
            if (e) {{
                e.preventDefault();
                e.stopPropagation();
                e.stopImmediatePropagation();
            }}
            
            if (isPolling) {{
                console.warn('⚠️ Translation already in progress');
                return false;
            }}
            
            const translateLink = getElement('{auto_translate_link_id}');
            if (!translateLink) {{
                console.error('❌ Auto translate link not found');
                return false;
            }}
            
            console.log('✅ Auto translate link found, starting background job...');
            
            // Get settings from Streamlit inputs
            let ollamaModel = 'aya:8b';
            let ollamaUrl = 'http://localhost:11434';
            let useRefinement = true;
            let temperature = 0.2;
            
            // Find Ollama Model input
            const modelInputs = document.querySelectorAll('input[type="text"]');
            for (let input of modelInputs) {{
                const closest = input.closest('.stTextInput');
                const label = closest ? (closest.querySelector('label') ? closest.querySelector('label').textContent : '') : '';
                if (label.includes('Ollama Model') || input.value.includes('aya')) {{
                    ollamaModel = input.value || 'aya:8b';
                    break;
                }}
            }}
            
            // Find Ollama Base URL input
            for (let input of modelInputs) {{
                const closest = input.closest('.stTextInput');
                const label = closest ? (closest.querySelector('label') ? closest.querySelector('label').textContent : '') : '';
                if (label.includes('Ollama Base URL') || (input.value.includes('localhost') && input.value.includes('11434'))) {{
                    ollamaUrl = input.value || 'http://localhost:11434';
                    break;
                }}
            }}
            
            // Find refinement checkbox
            const checkboxes = document.querySelectorAll('input[type="checkbox"]');
            for (let checkbox of checkboxes) {{
                const labelEl = checkbox.closest('label');
                const label = labelEl ? labelEl.textContent : '';
                if (label.includes('Two-Step') || label.includes('Refinement')) {{
                    useRefinement = checkbox.checked !== false;
                    break;
                }}
            }}
            
            // Find temperature slider
            const sliders = document.querySelectorAll('input[type="range"]');
            for (let slider of sliders) {{
                const closest = slider.closest('.stSlider');
                const label = closest ? (closest.querySelector('label') ? closest.querySelector('label').textContent : '') : '';
                if (label.includes('Temperature')) {{
                    temperature = parseFloat(slider.value || '0.2');
                    break;
                }}
            }}
            
            console.log('📋 Starting translation job with settings:', {{
                ollamaModel: ollamaModel,
                ollamaUrl: ollamaUrl,
                useRefinement: useRefinement,
                temperature: temperature
            }});
            
            // Show initial progress
            updateProgress({{
                status: 'pending',
                total_pages: totalPages,
                completed_pages: 0,
                failed_pages: 0,
                avg_time_per_page: 0,
                estimated_time_remaining: 0
            }});
            
            translateLink.textContent = '⏳ Starting...';
            translateLink.style.pointerEvents = 'none';
            translateLink.style.color = '#ff9800';
            
            // Start background job
            fetch(API_BASE + '/api/pdf/start-translation-job', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    book_id: bookId,
                    ollama_model: ollamaModel,
                    ollama_base_url: ollamaUrl,
                    use_refinement: useRefinement,
                    temperature: temperature,
                    parallel_workers: 10
                }})
            }})
            .then(response => response.json())
            .then(data => {{
                if (data.success) {{
                    jobId = data.job_id;
                    console.log('✅ Translation job started:', jobId);
                    translateLink.textContent = '⏳ Translating...';
                    isPolling = true;
                    // Start polling
                    pollJobProgress();
                }} else {{
                    console.error('❌ Failed to start job:', data.message);
                    translateLink.textContent = '❌ Failed to start';
                    translateLink.style.color = '#f44336';
                }}
            }})
            .catch(error => {{
                console.error('❌ Error starting job:', error);
                translateLink.textContent = '❌ Error';
                translateLink.style.color = '#f44336';
            }});
            
            return false;
        }}
        
        // Check for existing job on page load and automatically show progress dialog
        (function checkExistingJob() {{
            fetch(API_BASE + '/api/pdf/translation-job-for-book/' + bookId)
                .then(response => {{
                    if (response.status === 404) return null;
                    if (!response.ok) throw new Error('HTTP error! status: ' + response.status);
                    return response.json();
                }})
                .then(data => {{
                    if (data && (data.status === 'running' || data.status === 'pending')) {{
                        console.log('🔄 Found existing job:', data.job_id, 'Status:', data.status);
                        jobId = data.job_id;
                        isPolling = true;
                        // Automatically show progress dialog
                        updateProgress(data);
                        // Start polling immediately
                        pollJobProgress();
                    }} else if (data && data.status === 'stuck') {{
                        // Show stuck job with resume button
                        console.log('⚠️ Found stuck job:', data.job_id);
                        jobId = data.job_id;
                        updateProgress(data);
                        // Don't poll for stuck jobs - wait for user to resume
                    }} else if (data && (data.status === 'completed' || data.status === 'failed')) {{
                        // Show final status even if completed
                        console.log('✅ Found completed/failed job:', data.job_id);
                        jobId = data.job_id;
                        updateProgress(data);
                        // Don't poll for completed/failed jobs
                    }}
                }})
                .catch(error => {{
                    console.log('No existing job found or error:', error);
                }});
        }})();
        
        // Make function globally accessible
        window.{auto_translate_func_name} = {auto_translate_func_name};
        
        // Attach event listener
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', function() {{
                const link = getElement('{auto_translate_link_id}');
                if (link) {{
                    link.addEventListener('click', function(e) {{
                        e.preventDefault();
                        e.stopPropagation();
                        {auto_translate_func_name}(e);
                        return false;
                    }}, true);
                }}
            }});
        }} else {{
            const link = getElement('{auto_translate_link_id}');
            if (link) {{
                link.addEventListener('click', function(e) {{
                    e.preventDefault();
                    e.stopPropagation();
                    {auto_translate_func_name}(e);
                    return false;
                }}, true);
            }}
        }}
    }})();
    </script>
    <a href="#" 
       id="{auto_translate_link_id}" 
       onclick="event.preventDefault(); event.stopPropagation(); return {auto_translate_func_name}(event);"
       style="text-decoration: none; color: #1f77b4; cursor: pointer; font-size: 1em; display: inline-block; font-weight: bold; padding: 8px 16px; border: 2px solid #1f77b4; border-radius: 4px; background-color: #f0f8ff; z-index: 1000; position: relative;">
       🤖 Auto Translate
    </a>
    </div>
    """
    from streamlit.components.v1 import html
    html(auto_translate_html, height=50)

# Auto Translate settings (always visible)
with st.expander("⚙️ Auto Translation Settings", expanded=False):
    col_set1, col_set2, col_set3 = st.columns(3)
    with col_set1:
        ollama_model = st.text_input("Ollama Model", value="aya:8b", help="Model to use for translation", key="auto_ollama_model")
    with col_set2:
        ollama_base_url_input = st.text_input("Ollama Base URL", value=DEFAULT_OLLAMA_BASE_URL, key="auto_ollama_url")
    with col_set3:
        use_refinement = st.checkbox("Use Two-Step Refinement", value=True, key="auto_refinement")
        translation_temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1, key="auto_temp")
    
    st.info("💡 Click '🤖 Auto Translate' above to start. The process will run in the background on the server. You can close the browser and come back later - progress will be saved!")

# Navigation controls - placed early for quick response
col_nav1, col_nav2, col_nav3, col_nav4, col_nav5 = st.columns([1, 1, 2, 1, 1])

# Store current translation key
translation_key = f"translation_{book_id}_{current_page}"

def save_current_translation():
    """Save current translation if it exists in session state."""
    if translation_key in st.session_state:
        current_translation = st.session_state[translation_key]
        if current_translation and current_translation.strip():
            # Try to get page text if available (optional, for reference)
            page_text_cache_key = f"page_text_{book_id}_{current_page}"
            page_text = None
            if page_text_cache_key in st.session_state:
                page_text = st.session_state[page_text_cache_key]
            
            # Save translation (non-blocking, quick operation)
            # Note: We don't require page_text since we're displaying the page as image
            try:
                books_db.save_translation(
                    book_id=book_id,
                    page_number=current_page,
                    original_text=page_text if page_text else None,
                    translated_text=current_translation
                )
            except Exception as e:
                logger.error(f"Error saving translation: {e}")

with col_nav1:
    prev_clicked = st.button("◀️ Previous", disabled=(current_page == 0), use_container_width=True)
    if prev_clicked:
        save_current_translation()
        current_page_key = f'current_page_{book_id}'
        st.session_state[current_page_key] = max(0, current_page - 1)
        # Clear cached page data for old page
        if f"page_text_{book_id}_{current_page}" in st.session_state:
            del st.session_state[f"page_text_{book_id}_{current_page}"]
        if f"page_image_{book_id}_{current_page}" in st.session_state:
            del st.session_state[f"page_image_{book_id}_{current_page}"]
        st.rerun()

with col_nav2:
    next_clicked = st.button("Next ▶️", disabled=(current_page >= total_pages - 1), use_container_width=True)
    if next_clicked:
        save_current_translation()
        current_page_key = f'current_page_{book_id}'
        st.session_state[current_page_key] = min(total_pages - 1, current_page + 1)
        # Clear cached page data for old page
        if f"page_text_{book_id}_{current_page}" in st.session_state:
            del st.session_state[f"page_text_{book_id}_{current_page}"]
        if f"page_image_{book_id}_{current_page}" in st.session_state:
            del st.session_state[f"page_image_{book_id}_{current_page}"]
        st.rerun()

with col_nav3:
    st.markdown(f"<div style='text-align: center; padding-top: 10px;'><strong>Page {current_page + 1} of {total_pages}</strong></div>", unsafe_allow_html=True)

with col_nav4:
    # Jump to page input
    jump_page = st.number_input(
        "Go to page",
        min_value=1,
        max_value=total_pages,
        value=current_page + 1,
        key="jump_page_input",
        label_visibility="collapsed"
    )
    if jump_page != current_page + 1:
        save_current_translation()
        new_page = jump_page - 1
        # Clear cached page data for old page
        if f"page_text_{book_id}_{current_page}" in st.session_state:
            del st.session_state[f"page_text_{book_id}_{current_page}"]
        if f"page_image_{book_id}_{current_page}" in st.session_state:
            del st.session_state[f"page_image_{book_id}_{current_page}"]
        current_page_key = f'current_page_{book_id}'
        st.session_state[current_page_key] = new_page
        st.rerun()

with col_nav5:
    if st.button("🏠 Back to Books", use_container_width=True):
        # Clean up cached PDF document when leaving
        if pdf_cache_key in st.session_state:
            cached_doc = st.session_state.get(pdf_cache_key)
            if cached_doc is not None:
                try:
                    cached_doc.close()
                except:
                    pass
            del st.session_state[pdf_cache_key]
        st.switch_page("pages/10_📖_Books_Management.py")

st.markdown("---")

# Cache page image for better performance
page_image_cache_key = f"page_image_{book_id}_{current_page}"
cached_doc = st.session_state.get(pdf_cache_key)

# Two-panel layout
col_left, col_right = st.columns(2)

# Left panel: Original book page (displayed as image)
with col_left:
    col_title, col_copy_main, col_copy_prev = st.columns([2, 1, 1])
    with col_title:
        st.markdown("### 📄 Original Page")
    with col_copy_main:
        # Create JavaScript function first, then the link with inline onclick
        copy_link_id = f"copy_link_{book_id}_{current_page}"
        func_name = f"copyPdfText_{book_id}_{current_page}"
        
        # Create JavaScript function and link in one HTML block using html component
        copy_html = f"""
        <div>
        <script>
        function {func_name}(e, includePrev) {{
            if (e) {{
                e.preventDefault();
                e.stopPropagation();
            }}
            
            includePrev = includePrev || false;
            
            const copyLink = document.getElementById('{copy_link_id}');
            if (!copyLink) {{
                console.error('Copy link not found');
                return false;
            }}
            
            // Show loading state
            const originalText = copyLink.textContent;
            copyLink.textContent = '⏳ Copying...';
            copyLink.style.pointerEvents = 'none';
            copyLink.style.color = '#ff9800';
            
            console.log('Calling API for book_id={book_id}, page_number={current_page}');
            
            // Get API URL (fallback to default if not available)
            const apiUrl = window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1') 
                ? 'http://127.0.0.1:8000/api/pdf/extract-text'
                : 'http://127.0.0.1:8000/api/pdf/extract-text';
            
            console.log('API URL:', apiUrl);
            
            // Call API endpoint with include_next_page to complete sentences
            fetch(apiUrl, {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify({{
                    book_id: {book_id},
                    page_number: {current_page},
                    include_next_page: true,
                    include_previous_page: false
                }})
            }})
            .then(response => {{
                console.log('Response status:', response.status);
                if (!response.ok) {{
                    throw new Error('HTTP error! status: ' + response.status);
                }}
                return response.json();
            }})
            .then(data => {{
                console.log('Response data:', data);
                if (data.success && data.text) {{
                    // Copy to clipboard
                    const text = data.text;
                    let successMsg = '✅ Copied!';
                    if (data.includes_next_page) {{
                        successMsg += ' (includes next page to complete sentence)';
                    }}
                    
                    if (navigator.clipboard && navigator.clipboard.writeText) {{
                        navigator.clipboard.writeText(text).then(function() {{
                            copyLink.textContent = successMsg;
                            copyLink.style.color = '#28a745';
                            setTimeout(function() {{
                                copyLink.textContent = originalText;
                                copyLink.style.color = '#1f77b4';
                                copyLink.style.pointerEvents = 'auto';
                            }}, 3000);
                        }}, function(err) {{
                            console.error('Clipboard write failed:', err);
                            fallbackCopy(text, copyLink, originalText, successMsg);
                        }});
                    }} else {{
                        fallbackCopy(text, copyLink, originalText, successMsg);
                    }}
                }} else {{
                    alert('No text found on this page. The page might be image-only.');
                    copyLink.textContent = originalText;
                    copyLink.style.color = '#1f77b4';
                    copyLink.style.pointerEvents = 'auto';
                }}
            }})
            .catch(error => {{
                console.error('Fetch error:', error);
                console.error('Error name:', error.name);
                console.error('Error message:', error.message);
                let errorMsg = 'Failed to copy text. ';
                if (error.message && (error.message.includes('Failed to fetch') || error.message.includes('NetworkError'))) {{
                    errorMsg += '\\n\\nAPI server is not running or not accessible.\\n';
                    errorMsg += 'Please make sure the API server is running at http://127.0.0.1:8000\\n';
                    errorMsg += 'You can check the API server status in the sidebar of the main page.';
                }} else {{
                    errorMsg += error.message || 'Unknown error occurred';
                }}
                alert(errorMsg);
                copyLink.textContent = originalText;
                copyLink.style.color = '#1f77b4';
                copyLink.style.pointerEvents = 'auto';
            }});
            
            function fallbackCopy(text, link, originalText, successMsg) {{
                successMsg = successMsg || '✅ Copied!';
                const textArea = document.createElement('textarea');
                textArea.value = text;
                textArea.style.position = 'fixed';
                textArea.style.left = '-999999px';
                textArea.style.top = '-999999px';
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();
                try {{
                    const successful = document.execCommand('copy');
                    if (successful) {{
                        link.textContent = successMsg;
                        link.style.color = '#28a745';
                        setTimeout(function() {{
                            link.textContent = originalText;
                            link.style.color = '#1f77b4';
                            link.style.pointerEvents = 'auto';
                        }}, 3000);
                    }} else {{
                        alert('Failed to copy. Please try again.');
                        link.textContent = originalText;
                        link.style.pointerEvents = 'auto';
                    }}
                }} catch (err) {{
                    console.error('Fallback copy error:', err);
                    alert('Failed to copy. Please try again.');
                    link.textContent = originalText;
                    link.style.pointerEvents = 'auto';
                }}
                document.body.removeChild(textArea);
            }}
            
            return false;
        }}
        </script>
        <a href="javascript:void(0)" 
           id="{copy_link_id}" 
           onclick="return {func_name}(event, false);"
           style="text-decoration: none; color: #1f77b4; cursor: pointer; font-size: 0.9em; display: inline-block;">
           📋 Copy Text
        </a>
        </div>
        """
        from streamlit.components.v1 import html
        html(copy_html, height=30)
    
    with col_copy_prev:
            # Copy from previous page link (only show if not first page)
            if current_page > 0:
                copy_prev_link_id = f"copy_prev_link_{book_id}_{current_page}"
                func_prev_name = f"copyPdfTextPrev_{book_id}_{current_page}"
                
                copy_prev_html = f"""
                <div>
                <script>
                function {func_prev_name}(e) {{
                    if (e) {{
                        e.preventDefault();
                        e.stopPropagation();
                    }}
                    
                    const copyLink = document.getElementById('{copy_prev_link_id}');
                    if (!copyLink) return false;
                    
                    const originalText = copyLink.textContent;
                    copyLink.textContent = '⏳ Copying...';
                    copyLink.style.pointerEvents = 'none';
                    
                    fetch('http://127.0.0.1:8000/api/pdf/extract-text', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            book_id: {book_id},
                            page_number: {current_page - 1},
                            include_next_page: true,
                            include_previous_page: false
                        }})
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        if (data.success && data.text) {{
                            const text = data.text;
                            if (navigator.clipboard && navigator.clipboard.writeText) {{
                                navigator.clipboard.writeText(text).then(() => {{
                                    copyLink.textContent = '✅ Copied!';
                                    copyLink.style.color = '#28a745';
                                    setTimeout(() => {{
                                        copyLink.textContent = originalText;
                                        copyLink.style.color = '#1f77b4';
                                        copyLink.style.pointerEvents = 'auto';
                                    }}, 2000);
                                }});
                            }}
                        }} else {{
                            alert('No text found.');
                            copyLink.textContent = originalText;
                            copyLink.style.pointerEvents = 'auto';
                        }}
                    }})
                    .catch(error => {{
                        alert('Failed to copy: ' + error.message);
                        copyLink.textContent = originalText;
                        copyLink.style.pointerEvents = 'auto';
                    }});
                    
                    return false;
                }}
                </script>
                <a href="javascript:void(0)" 
                   id="{copy_prev_link_id}" 
                   onclick="return {func_prev_name}(event);"
                   style="text-decoration: none; color: #666; cursor: pointer; font-size: 0.8em; display: inline-block;"
                   title="Copy from previous page to complete sentence">
                   ↶ Prev Page
                </a>
                </div>
                """
                html(copy_prev_html, height=25)
    
    # Render and display PDF page as image
    if page_image_cache_key in st.session_state:
        page_image = st.session_state[page_image_cache_key]
    else:
        # Render page as image (use cached document for speed)
        # Use zoom 2.0 for good quality and readability
        with st.spinner("Loading page..."):
            page_image = render_page_as_image(pdf_path, current_page, zoom=2.0, cached_doc=cached_doc)
            if page_image:
                st.session_state[page_image_cache_key] = page_image
    
    if page_image:
        # Display the PDF page image
        # Render at 2.0x zoom for good quality, display with container width
        # Users can use browser zoom (Ctrl/Cmd +) for closer inspection
        st.image(page_image, use_container_width=True)
        st.caption(f"Page {current_page + 1} of {total_pages} | 💡 Tip: Use browser zoom (Ctrl/Cmd +) for closer view")
        
        # Optional: Show extracted text in expander for reference (if available)
        page_text_cache_key = f"page_text_{book_id}_{current_page}"
        if page_text_cache_key not in st.session_state:
            # Try to extract text for reference (optional, non-blocking)
            try:
                page_text = extract_page_text(pdf_path, current_page, cached_doc=cached_doc)
                if page_text and page_text.strip():
                    st.session_state[page_text_cache_key] = page_text
            except:
                pass
        
        # Show extracted text in expander if available
        if page_text_cache_key in st.session_state:
            page_text = st.session_state[page_text_cache_key]
            if page_text and page_text.strip():
                with st.expander("📝 Extracted Text (for reference)", expanded=False):
                    formatted_text = format_text_for_display(page_text)
                    st.text_area(
                        "Extracted Text",
                        value=formatted_text,
                        height=200,
                        disabled=True,
                        key="extracted_text_reference",
                        label_visibility="collapsed"
                    )
    else:
        st.error("⚠️ Could not render PDF page. The PDF file may be corrupted.")
        st.text_area(
            "Original Page",
            value="",
            height=600,
            disabled=True,
            key="original_page_error",
            label_visibility="collapsed"
        )

# Right panel: Translation input
with col_right:
    st.markdown("### ✍️ Translation")
    
    # Preview toggle button
    preview_mode_key = f"preview_mode_{book_id}_{current_page}"
    if preview_mode_key not in st.session_state:
        st.session_state[preview_mode_key] = False
    
    col_preview_toggle, col_preview_info = st.columns([1, 3])
    with col_preview_toggle:
        preview_mode = st.button(
            "👁️ Preview" if not st.session_state[preview_mode_key] else "✏️ Edit",
            key=f"preview_toggle_{book_id}_{current_page}",
            use_container_width=True
        )
        if preview_mode:
            st.session_state[preview_mode_key] = not st.session_state[preview_mode_key]
            st.rerun()
    
    with col_preview_info:
        if st.session_state[preview_mode_key]:
            st.info("👁️ Preview mode: Showing formatted version")
        else:
            st.info("✏️ Edit mode: Type your translation here")
    
    # Load existing translation if available
    translation_data = books_db.get_translation(book_id, current_page)
    existing_translation = ""
    if translation_data and translation_data.get('translated_text'):
        existing_translation = translation_data['translated_text']
    
    # Translation input area - formatted similarly to original
    translation_textarea_key = f"translation_{book_id}_{current_page}"
    translation_textarea_id = f"translation_textarea_{book_id}_{current_page}"
    
    # Add JavaScript to convert pasted HTML to markdown
    paste_converter_js = f"""
    <script>
    (function() {{
        // Simple HTML to Markdown converter
        function htmlToMarkdown(html) {{
            if (!html) return '';
            
            // Create a temporary div to parse HTML
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = html;
            
            function convertNode(node) {{
                if (node.nodeType === Node.TEXT_NODE) {{
                    return node.textContent;
                }}
                
                if (node.nodeType !== Node.ELEMENT_NODE) {{
                    return '';
                }}
                
                const tagName = node.tagName.toLowerCase();
                const children = Array.from(node.childNodes).map(convertNode).join('');
                
                switch(tagName) {{
                    case 'p':
                        return children + '\\n\\n';
                    case 'br':
                        return '\\n';
                    case 'strong':
                    case 'b':
                        return '**' + children + '**';
                    case 'em':
                    case 'i':
                        return '*' + children + '*';
                    case 'h1':
                        return '# ' + children + '\\n\\n';
                    case 'h2':
                        return '## ' + children + '\\n\\n';
                    case 'h3':
                        return '### ' + children + '\\n\\n';
                    case 'h4':
                        return '#### ' + children + '\\n\\n';
                    case 'ul':
                        return children + '\\n';
                    case 'ol':
                        return children + '\\n';
                    case 'li':
                        return '- ' + children + '\\n';
                    case 'blockquote':
                        return '> ' + children + '\\n\\n';
                    case 'code':
                        return '`' + children + '`';
                    case 'pre':
                        return '```\\n' + children + '\\n```\\n\\n';
                    case 'a':
                        const href = node.getAttribute('href') || '';
                        return '[' + children + '](' + href + ')';
                    default:
                        return children;
                }}
            }}
            
            return convertNode(tempDiv).trim();
        }}
        
        // Wait for textarea to be available
        function setupPasteHandler() {{
            const columns = document.querySelectorAll('[data-testid="column"]');
            if (columns.length >= 2) {{
                const rightColumn = columns[1];
                const textarea = rightColumn.querySelector('textarea');
                if (textarea) {{
                    textarea.id = '{translation_textarea_id}';
                    
                    // Handle paste event
                    textarea.addEventListener('paste', function(e) {{
                        const clipboardData = e.clipboardData || window.clipboardData;
                        const pastedData = clipboardData.getData('text/html') || clipboardData.getData('text/plain');
                        
                        if (pastedData && pastedData.includes('<')) {{
                            // HTML content detected, convert to markdown
                            e.preventDefault();
                            e.stopPropagation();
                            const markdown = htmlToMarkdown(pastedData);
                            
                            // Insert markdown at cursor position
                            const start = textarea.selectionStart;
                            const end = textarea.selectionEnd;
                            const text = textarea.value;
                            const newText = text.substring(0, start) + markdown + text.substring(end);
                            textarea.value = newText;
                            
                            // Set cursor position after inserted text
                            const newCursorPos = start + markdown.length;
                            textarea.setSelectionRange(newCursorPos, newCursorPos);
                            
                            // Don't trigger any events that would cause Streamlit rerun
                            // The value will be saved manually when user clicks save button
                            console.log('Converted HTML to Markdown:', markdown.substring(0, 100));
                        }}
                    }}, true); // Use capture phase to intercept early
                    
                    // Prevent Streamlit from detecting changes on blur
                    // This prevents auto-save and page refresh
                    let originalValue = textarea.value;
                    textarea.addEventListener('focus', function() {{
                        originalValue = textarea.value;
                    }});
                    
                    textarea.addEventListener('blur', function(e) {{
                        // If value hasn't actually changed, prevent any rerun
                        if (textarea.value === originalValue) {{
                            e.stopPropagation();
                        }}
                        // Note: Streamlit will still rerun if value changed, but at least we prevent unnecessary reruns
                        console.log('Textarea blurred - value will be saved only when you click Save button');
                    }});
                    
                    console.log('Paste handler set up for translation textarea');
                }} else {{
                    // Retry after a short delay
                    setTimeout(setupPasteHandler, 100);
                }}
            }} else {{
                setTimeout(setupPasteHandler, 100);
            }}
        }}
        
        // Start setup when DOM is ready
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', setupPasteHandler);
        }} else {{
            setupPasteHandler();
        }}
    }})();
    </script>
    """
    from streamlit.components.v1 import html
    html(paste_converter_js, height=0)
    
    # Show either textarea or preview based on mode
    if st.session_state[preview_mode_key]:
        # Preview mode: Show formatted markdown
        # Get current text from session state or existing translation
        current_text = st.session_state.get(translation_textarea_key, existing_translation)
        if not current_text:
            current_text = existing_translation
        
        # Initialize translation_text for consistency
        translation_text = current_text
        
        if current_text:
            st.markdown("---")
            st.markdown("### 📄 Formatted Preview")
            # Render markdown with proper styling
            st.markdown(current_text)
            st.markdown("---")
        else:
            st.info("No translation text to preview. Switch to Edit mode to add content.")
    else:
        # Edit mode: Use st.form to prevent auto-refresh on blur
        # This prevents Streamlit from rerunning when textarea loses focus
        with st.form(key=f"translation_form_{book_id}_{current_page}", clear_on_submit=False, border=False):
            # Store the textarea ID in a data attribute for JavaScript to find
            translation_text = st.text_area(
                "Enter your translation",
                value=existing_translation,
                height=600,
                key=translation_textarea_key,
                label_visibility="collapsed",
                help="Enter the translation for this page. Paste formatted text and it will be automatically converted to Markdown format. Click 'Save Translation' below to save. The page won't refresh when you click away."
            )
            
            # Add JavaScript to set the textarea ID after form renders
            set_id_js = f"""
            <script>
            (function() {{
                function setTextareaId() {{
                    const forms = document.querySelectorAll('form');
                    for (let form of forms) {{
                        const textarea = form.querySelector('textarea');
                        if (textarea && !textarea.id) {{
                            textarea.id = '{translation_textarea_id}';
                            textarea.setAttribute('data-translation-key', '{translation_textarea_key}');
                            console.log('Set textarea ID:', '{translation_textarea_id}');
                            break;
                        }}
                    }}
                }}
                // Try immediately
                setTextareaId();
                // Also try after a short delay in case form hasn't rendered yet
                setTimeout(setTextareaId, 100);
                setTimeout(setTextareaId, 500);
            }})();
            </script>
            """
            from streamlit.components.v1 import html
            html(set_id_js, height=0)
            
            # Form submit button - but we'll handle save via the save link outside the form
            # This form just prevents auto-refresh, we don't use the submit button
            form_submitted = st.form_submit_button("💾 Save Translation", use_container_width=True, type="primary")
            
            # If form is submitted via button, save the translation via API
            if form_submitted:
                try:
                    response = requests.post(
                        'http://127.0.0.1:8000/api/pdf/save-translation',
                        json={
                            'book_id': book_id,
                            'page_number': current_page,
                            'translated_text': translation_text,
                            'original_text': None
                        },
                        timeout=5
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('success'):
                            st.success("✅ Translation saved successfully!")
                            # Update session state to reflect saved translation
                            st.session_state[translation_textarea_key] = translation_text
                        else:
                            st.error(f"Failed to save: {data.get('message', 'Unknown error')}")
                    else:
                        st.error(f"Failed to save: HTTP {response.status_code}")
                except Exception as e:
                    st.error(f"Error saving translation: {str(e)}")
    
    # Save link and status
    col_save1, col_save2 = st.columns([1, 1])
    
    # Get current translation text for save functionality
    # If in form, get from session state; otherwise use the variable directly
    if not st.session_state[preview_mode_key]:
        # In edit mode, get from session state (form updates session state)
        current_translation_text = st.session_state.get(translation_textarea_key, existing_translation)
    else:
        # In preview mode, use the variable
        current_translation_text = translation_text
    
    # Check if there are unsaved changes
    has_unsaved_changes = current_translation_text != existing_translation
    
    with col_save1:
        # Create save translation hyperlink with JavaScript
        save_link_id = f"save_link_{book_id}_{current_page}"
        save_func_name = f"saveTranslation_{book_id}_{current_page}"
        
        save_html = f"""
        <script>
        (function() {{
            function {save_func_name}(e) {{
                if (e) {{
                    e.preventDefault();
                    e.stopPropagation();
                    e.stopImmediatePropagation();
                }}
                
                const saveLink = document.getElementById('{save_link_id}');
                if (!saveLink) {{
                    console.error('Save link not found');
                    return false;
                }}
                
            // Get translation text from textarea
            // Since textarea is inside a form, we need to search more carefully
            let translationText = '';
            
            // Try multiple strategies to find the textarea
            // Strategy 1: Find by ID (if we set one)
            const textareaById = document.getElementById('{translation_textarea_id}');
            if (textareaById) {{
                translationText = textareaById.value || '';
                console.log('Found textarea by ID, length:', translationText.length);
            }}
            
            // Strategy 2: Find in form (textarea is inside a form)
            if (!translationText) {{
                const forms = document.querySelectorAll('form');
                for (let form of forms) {{
                    const textarea = form.querySelector('textarea');
                    if (textarea && textarea.value) {{
                        translationText = textarea.value || '';
                        console.log('Found textarea in form, length:', translationText.length);
                        break;
                    }}
                }}
            }}
            
            // Strategy 3: Find in right column
            if (!translationText) {{
                const columns = document.querySelectorAll('[data-testid="column"]');
                if (columns.length >= 2) {{
                    const rightColumn = columns[1];
                    const textarea = rightColumn.querySelector('textarea');
                    if (textarea) {{
                        translationText = textarea.value || '';
                        console.log('Found textarea in right column, length:', translationText.length);
                    }}
                }}
            }}
            
            // Strategy 4: Try all textareas (last resort)
            if (!translationText) {{
                const textareas = document.querySelectorAll('textarea');
                for (let ta of textareas) {{
                    if (ta.value && ta.value.trim().length > 0) {{
                        translationText = ta.value || '';
                        console.log('Found textarea (fallback), length:', translationText.length);
                        break;
                    }}
                }}
            }}
            
            // If still no text, log error
            if (!translationText) {{
                console.warn('Could not find textarea with content. Using fallback value.');
                translationText = {json.dumps(current_translation_text) if current_translation_text else '""'};
            }}
            
            console.log('Final translation text length:', translationText.length);
                
                console.log('Translation text length:', translationText.length);
                console.log('Saving translation for book_id={book_id}, page_number={current_page}');
                
                // Show loading state
                const originalText = saveLink.textContent;
                saveLink.textContent = '⏳ Saving...';
                saveLink.style.pointerEvents = 'none';
                saveLink.style.color = '#ff9800';
                
                // Call API endpoint to save translation
                fetch('http://127.0.0.1:8000/api/pdf/save-translation', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{
                        book_id: {book_id},
                        page_number: {current_page},
                        translated_text: translationText,
                        original_text: null
                    }})
                }})
                .then(response => {{
                    console.log('Save response status:', response.status);
                    if (!response.ok) {{
                        throw new Error('HTTP error! status: ' + response.status);
                    }}
                    return response.json();
                }})
                .then(data => {{
                    console.log('Save response data:', data);
                    console.log('Translation text length sent:', translationText.length);
                    console.log('Translation text preview:', translationText.substring(0, 100));
                    if (data.success) {{
                        console.log('✅ Translation saved successfully!');
                        saveLink.textContent = '✅ Saved!';
                        saveLink.style.color = '#28a745';
                        setTimeout(function() {{
                            saveLink.textContent = originalText;
                            saveLink.style.color = '#1f77b4';
                            saveLink.style.pointerEvents = 'auto';
                        }}, 2000);
                    }} else {{
                        console.error('❌ Save failed:', data);
                        alert('Failed to save translation: ' + (data.message || 'Unknown error'));
                        saveLink.textContent = originalText;
                        saveLink.style.color = '#1f77b4';
                        saveLink.style.pointerEvents = 'auto';
                    }}
                }})
                .catch(error => {{
                    console.error('❌ Save error:', error);
                    console.error('Error stack:', error.stack);
                    console.error('Translation text that failed:', translationText.substring(0, 100));
                    alert('Failed to save translation: ' + error.message + '. Please check if API server is running at http://127.0.0.1:8000');
                    saveLink.textContent = originalText;
                    saveLink.style.color = '#1f77b4';
                    saveLink.style.pointerEvents = 'auto';
                }});
                
                return false;
            }}
            
            // Make function globally accessible
            window.{save_func_name} = {save_func_name};
        }})();
        </script>
        <a href="javascript:void(0)" 
           id="{save_link_id}" 
           onclick="return window.{save_func_name}(event);"
           style="text-decoration: none; color: #1f77b4; cursor: pointer; font-size: 0.9em; display: inline-block; font-weight: bold;">
           💾 Save Translation
        </a>
        """
        from streamlit.components.v1 import html
        html(save_html, height=30)
    
    with col_save2:
        # Show save status
        if has_unsaved_changes:
            st.warning("💡 Unsaved changes")
        elif existing_translation:
            st.success("✅ Saved")
        else:
            st.info("📝 No translation yet")

# Show translation progress
st.markdown("---")
translated_pages = books_db.get_translated_pages(book_id)
translation_progress = len(translated_pages) / total_pages * 100 if total_pages > 0 else 0

col_prog1, col_prog2 = st.columns([3, 1])
with col_prog1:
    st.progress(translation_progress / 100)
with col_prog2:
    st.markdown(f"**{len(translated_pages)}/{total_pages} pages translated ({translation_progress:.1f}%)**")

# Show which pages have translations
if translated_pages:
    st.caption(f"Translated pages: {', '.join([str(p + 1) for p in translated_pages])}")

