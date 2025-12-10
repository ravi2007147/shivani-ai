"""Books Management Page - Upload and manage PDF books.

This page allows you to:
- Upload PDF files
- View list of uploaded books with previews
- Manage your book collection
"""

import streamlit as st
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import os
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.books_db import BooksDBManager

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Books Management",
    page_icon="📖",
    layout="wide"
)

st.title("📖 Books Management")
st.markdown("Upload and manage your PDF book collection with previews")


def get_books_storage_dir() -> Path:
    """Get the directory where books are stored."""
    books_dir = project_root / ".books_storage"
    books_dir.mkdir(exist_ok=True)
    return books_dir


def get_previews_storage_dir() -> Path:
    """Get the directory where preview images are stored."""
    previews_dir = project_root / ".books_previews"
    previews_dir.mkdir(exist_ok=True)
    return previews_dir


def generate_preview_image(pdf_path: Path, output_path: Path) -> bool:
    """Generate a preview image from the first page of a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path where the preview image should be saved
        
    Returns:
        True if preview was generated successfully, False otherwise
    """
    try:
        # Try PyMuPDF (fitz) first - best for rendering
        import fitz  # pymupdf
        
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            doc.close()
            return False
        
        # Get first page
        page = doc[0]
        
        # Render page to image (zoom factor for better quality)
        zoom = 2.0  # 2x zoom for better quality
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Save as PNG
        pix.save(output_path)
        doc.close()
        return True
        
    except ImportError:
        try:
            # Fallback to pypdf + PIL
            from pypdf import PdfReader
            from PIL import Image
            import io
            
            reader = PdfReader(pdf_path)
            if len(reader.pages) == 0:
                return False
            
            # Try to extract images from first page
            first_page = reader.pages[0]
            if '/XObject' in first_page['/Resources']:
                xObject = first_page['/Resources']['/XObject'].get_object()
                for obj in xObject:
                    if xObject[obj]['/Subtype'] == '/Image':
                        # Found an image, extract it
                        size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                        data = xObject[obj].get_data()
                        img = Image.open(io.BytesIO(data))
                        img.save(output_path)
                        return True
            
            # If no image found, create a placeholder
            img = Image.new('RGB', (400, 600), color='white')
            img.save(output_path)
            return True
            
        except ImportError:
            logger.warning("No PDF library available for preview generation")
            return False
    except Exception as e:
        logger.error(f"Error generating preview: {e}")
        return False


def save_uploaded_pdf(uploaded_file, books_db: BooksDBManager) -> Optional[int]:
    """Save an uploaded PDF file and add it to the database.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        books_db: BooksDBManager instance
        
    Returns:
        Book ID if successful, None otherwise
    """
    try:
        # Create storage directories
        books_dir = get_books_storage_dir()
        previews_dir = get_previews_storage_dir()
        
        # Generate unique filename to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in (' ', '-', '_', '.')).strip()
        unique_filename = f"{timestamp}_{safe_filename}"
        file_path = books_dir / unique_filename
        
        # Save the PDF file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        file_size = file_path.stat().st_size
        
        # Extract metadata if possible
        title = None
        author = None
        try:
            from src.utils.pdf_parser import extract_metadata_from_pdf_stream
            uploaded_file.seek(0)
            metadata = extract_metadata_from_pdf_stream(uploaded_file)
            title = metadata.get('Title')
            author = metadata.get('Author')
        except Exception as e:
            logger.warning(f"Could not extract metadata: {e}")
        
        # Generate preview image
        preview_filename = f"{timestamp}_{Path(safe_filename).stem}_preview.png"
        preview_path = previews_dir / preview_filename
        preview_generated = generate_preview_image(file_path, preview_path)
        
        preview_image_path = str(preview_path) if preview_generated else None
        
        # Add to database
        book_id = books_db.add_book(
            filename=uploaded_file.name,
            file_path=str(file_path),
            file_size=file_size,
            title=title,
            author=author,
            preview_image_path=preview_image_path
        )
        
        return book_id
        
    except Exception as e:
        logger.error(f"Error saving PDF: {e}")
        st.error(f"Error saving PDF: {e}")
        return None


# Initialize books database manager
books_db = BooksDBManager()

# Upload section
st.markdown("---")
st.markdown("### 📤 Upload PDF Book")

uploaded_file = st.file_uploader(
    "Choose a PDF file to upload",
    type=["pdf"],
    help="Upload a PDF book file. The first page will be used as a preview."
)

if uploaded_file is not None:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(f"**File:** {uploaded_file.name} ({len(uploaded_file.getbuffer()) / 1024:.2f} KB)")
    
    with col2:
        if st.button("📥 Upload Book", type="primary", use_container_width=True):
            with st.spinner("Uploading and processing PDF..."):
                book_id = save_uploaded_pdf(uploaded_file, books_db)
                if book_id:
                    st.success(f"✅ Book uploaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to upload book. Please try again.")

# Display uploaded books
st.markdown("---")
st.markdown("### 📚 Your Books Collection")

all_books = books_db.list_books()

if not all_books:
    st.info("📖 No books uploaded yet. Use the upload section above to add your first book!")
else:
    total_size = books_db.get_total_size()
    st.markdown(f"**Total books:** {len(all_books)} | **Total size:** {total_size / (1024*1024):.2f} MB")
    st.markdown("---")
    
    # Display books in a grid
    cols_per_row = 3
    for idx in range(0, len(all_books), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for col_idx, col in enumerate(cols):
            book_idx = idx + col_idx
            if book_idx < len(all_books):
                book = all_books[book_idx]
                
                with col:
                    # Create a container for each book
                    with st.container():
                        # Display preview image (clickable)
                        preview_path = book.get('preview_image_path')
                        if preview_path and os.path.exists(preview_path):
                            st.image(preview_path, use_container_width=True)
                        else:
                            # Placeholder for missing preview
                            st.markdown(
                                '<div style="background-color: #f0f0f0; padding: 60px 20px; text-align: center; border-radius: 5px;">'
                                '<span style="font-size: 48px;">📖</span><br>'
                                '<span style="color: #666;">No preview available</span>'
                                '</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Book information
                        title = book.get('title') or book.get('filename', 'Untitled')
                        author = book.get('author') or 'Unknown Author'
                        
                        st.markdown(f"**{title}**")
                        st.caption(f"by {author}")
                        st.caption(f"📄 {book.get('filename', 'N/A')}")
                        
                        # File size
                        file_size = book.get('file_size', 0)
                        if file_size > 1024 * 1024:
                            size_str = f"{file_size / (1024*1024):.2f} MB"
                        else:
                            size_str = f"{file_size / 1024:.2f} KB"
                        st.caption(f"📊 {size_str}")
                        
                        # Upload date
                        upload_date = book.get('upload_date')
                        if upload_date:
                            try:
                                if isinstance(upload_date, str):
                                    date_obj = datetime.fromisoformat(upload_date.replace('Z', '+00:00'))
                                else:
                                    date_obj = upload_date
                                st.caption(f"📅 {date_obj.strftime('%Y-%m-%d')}")
                            except:
                                pass
                        
                        # Action buttons
                        col_btn1, col_btn2 = st.columns(2)
                        
                        with col_btn1:
                            # Open/Translate button
                            if st.button("📖 Open", key=f"open_{book['id']}", use_container_width=True, type="primary"):
                                # Store book_id in session state and navigate
                                st.session_state.selected_book_id = book['id']
                                st.switch_page("pages/11_📖_Book_Translation.py")
                        
                        with col_btn2:
                            # Delete button
                            if st.button("🗑️ Delete", key=f"delete_{book['id']}", use_container_width=True):
                                if books_db.delete_book(book['id']):
                                    st.success("Book deleted!")
                                    st.rerun()
                                else:
                                    st.error("Failed to delete book")

