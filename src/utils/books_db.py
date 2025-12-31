"""Books Database Manager - SQLite database for managing uploaded PDF books.

This module provides a database manager for storing:
- Book metadata (title, author, filename, file path, etc.)
- Upload timestamps
- Preview image paths
"""

import sqlite3
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class BooksDBManager:
    """Manages SQLite database for PDF books."""
    
    def __init__(self, db_path: str = None):
        """Initialize the Books Database Manager.
        
        Args:
            db_path: Path to SQLite database file (default: .books_db/books.db)
        """
        if db_path is None:
            # Use project root directory
            project_root = Path(__file__).parent.parent.parent
            db_dir = project_root / ".books_db"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "books.db")
        
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create books table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                author TEXT,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL UNIQUE,
                file_size INTEGER,
                preview_image_path TEXT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on filename for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_books_filename ON books(filename)
        """)
        
        # Create book_translations table for storing page-by-page translations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS book_translations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL,
                page_number INTEGER NOT NULL,
                original_text TEXT,
                translated_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (book_id) REFERENCES books(id) ON DELETE CASCADE,
                UNIQUE(book_id, page_number)
            )
        """)
        
        # Create index on book_id and page_number for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_translations_book_page ON book_translations(book_id, page_number)
        """)
        
        # Create translation_jobs table for tracking background translation jobs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS translation_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                total_pages INTEGER NOT NULL,
                completed_pages INTEGER DEFAULT 0,
                failed_pages INTEGER DEFAULT 0,
                ollama_model TEXT,
                ollama_base_url TEXT,
                use_refinement BOOLEAN,
                temperature REAL,
                parallel_workers INTEGER DEFAULT 10,
                avg_time_per_page REAL,
                estimated_time_remaining REAL,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                FOREIGN KEY (book_id) REFERENCES books(id) ON DELETE CASCADE
            )
        """)
        
        # Create index on book_id and status for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_book_status ON translation_jobs(book_id, status)
        """)
        
        # Add parallel_workers column if it doesn't exist (migration)
        try:
            cursor.execute("ALTER TABLE translation_jobs ADD COLUMN parallel_workers INTEGER DEFAULT 10")
            conn.commit()
            logger.info("Added parallel_workers column to translation_jobs table")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower() or "already exists" in str(e).lower():
                # Column already exists, that's fine
                pass
            else:
                logger.warning(f"Could not add parallel_workers column: {e}")
        
        conn.commit()
        conn.close()
    
    def add_book(
        self,
        filename: str,
        file_path: str,
        file_size: int,
        title: Optional[str] = None,
        author: Optional[str] = None,
        preview_image_path: Optional[str] = None
    ) -> int:
        """Add a new book to the database.
        
        Args:
            filename: Original filename of the PDF
            file_path: Full path to the stored PDF file
            file_size: Size of the file in bytes
            title: Optional title extracted from PDF metadata
            author: Optional author extracted from PDF metadata
            preview_image_path: Optional path to preview image
            
        Returns:
            ID of the newly created book record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO books (title, author, filename, file_path, file_size, preview_image_path, upload_date, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                title,
                author,
                filename,
                file_path,
                file_size,
                preview_image_path,
                datetime.now(),
                datetime.now()
            ))
            
            book_id = cursor.lastrowid
            conn.commit()
            return book_id
        except sqlite3.IntegrityError:
            logger.warning(f"Book with file_path '{file_path}' already exists")
            conn.rollback()
            # Return existing book ID
            cursor.execute("SELECT id FROM books WHERE file_path = ?", (file_path,))
            result = cursor.fetchone()
            return result[0] if result else None
        finally:
            conn.close()
    
    def list_books(self, order_by: str = "upload_date DESC") -> List[Dict]:
        """List all books in the database.
        
        Args:
            order_by: SQL ORDER BY clause (default: upload_date DESC)
            
        Returns:
            List of dictionaries containing book information
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"""
                SELECT * FROM books ORDER BY {order_by}
            """)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def get_book(self, book_id: int) -> Optional[Dict]:
        """Get a specific book by ID.
        
        Args:
            book_id: ID of the book
            
        Returns:
            Dictionary containing book information, or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM books WHERE id = ?", (book_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
    
    def update_book(
        self,
        book_id: int,
        title: Optional[str] = None,
        author: Optional[str] = None,
        preview_image_path: Optional[str] = None
    ) -> bool:
        """Update book metadata.
        
        Args:
            book_id: ID of the book to update
            title: New title (optional)
            author: New author (optional)
            preview_image_path: New preview image path (optional)
            
        Returns:
            True if update was successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            updates = []
            params = []
            
            if title is not None:
                updates.append("title = ?")
                params.append(title)
            
            if author is not None:
                updates.append("author = ?")
                params.append(author)
            
            if preview_image_path is not None:
                updates.append("preview_image_path = ?")
                params.append(preview_image_path)
            
            if not updates:
                return False
            
            updates.append("updated_at = ?")
            params.append(datetime.now())
            params.append(book_id)
            
            cursor.execute(f"""
                UPDATE books SET {', '.join(updates)} WHERE id = ?
            """, params)
            
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def delete_book(self, book_id: int) -> bool:
        """Delete a book from the database.
        
        Args:
            book_id: ID of the book to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get file paths before deletion
            cursor.execute("SELECT file_path, preview_image_path FROM books WHERE id = ?", (book_id,))
            result = cursor.fetchone()
            
            if not result:
                return False
            
            file_path, preview_image_path = result
            
            # Delete from database
            cursor.execute("DELETE FROM books WHERE id = ?", (book_id,))
            conn.commit()
            
            # Delete files
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Could not delete file {file_path}: {e}")
            
            if preview_image_path and os.path.exists(preview_image_path):
                try:
                    os.remove(preview_image_path)
                except Exception as e:
                    logger.warning(f"Could not delete preview image {preview_image_path}: {e}")
            
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def get_total_size(self) -> int:
        """Get total size of all stored books in bytes.
        
        Returns:
            Total size in bytes
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT SUM(file_size) FROM books")
            result = cursor.fetchone()
            return result[0] if result[0] else 0
        finally:
            conn.close()
    
    def save_translation(
        self,
        book_id: int,
        page_number: int,
        original_text: Optional[str] = None,
        translated_text: Optional[str] = None
    ) -> bool:
        """Save or update translation for a specific page.
        
        Args:
            book_id: ID of the book
            page_number: Page number (0-indexed or 1-indexed, will be stored as-is)
            original_text: Original text from the page (optional)
            translated_text: Translated text (optional)
            
        Returns:
            True if save was successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if translation already exists
            cursor.execute(
                "SELECT id FROM book_translations WHERE book_id = ? AND page_number = ?",
                (book_id, page_number)
            )
            exists = cursor.fetchone()
            
            if exists:
                # Update existing translation
                updates = []
                params = []
                
                if original_text is not None:
                    updates.append("original_text = ?")
                    params.append(original_text)
                
                if translated_text is not None:
                    updates.append("translated_text = ?")
                    params.append(translated_text)
                
                if updates:
                    updates.append("updated_at = ?")
                    params.append(datetime.now())
                    params.append(book_id)
                    params.append(page_number)
                    
                    cursor.execute(f"""
                        UPDATE book_translations 
                        SET {', '.join(updates)} 
                        WHERE book_id = ? AND page_number = ?
                    """, params)
            else:
                # Insert new translation
                cursor.execute("""
                    INSERT INTO book_translations 
                    (book_id, page_number, original_text, translated_text, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    book_id,
                    page_number,
                    original_text,
                    translated_text,
                    datetime.now(),
                    datetime.now()
                ))
            
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving translation: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_translation(self, book_id: int, page_number: int) -> Optional[Dict]:
        """Get translation for a specific page.
        
        Args:
            book_id: ID of the book
            page_number: Page number
            
        Returns:
            Dictionary containing translation data, or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM book_translations 
                WHERE book_id = ? AND page_number = ?
            """, (book_id, page_number))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
    
    def get_all_translations(self, book_id: int) -> List[Dict]:
        """Get all translations for a book.
        
        Args:
            book_id: ID of the book
            
        Returns:
            List of dictionaries containing translation data
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM book_translations 
                WHERE book_id = ? 
                ORDER BY page_number
            """, (book_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def get_translated_pages(self, book_id: int) -> List[int]:
        """Get list of page numbers that have translations.
        
        Args:
            book_id: ID of the book
            
        Returns:
            List of page numbers that have translations
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT page_number FROM book_translations 
                WHERE book_id = ? AND translated_text IS NOT NULL AND translated_text != ''
                ORDER BY page_number
            """, (book_id,))
            rows = cursor.fetchall()
            return [row[0] for row in rows]
        finally:
            conn.close()
    
    def create_translation_job(
        self,
        book_id: int,
        total_pages: int,
        ollama_model: str,
        ollama_base_url: str,
        use_refinement: bool,
        temperature: float,
        parallel_workers: int = 10
    ) -> int:
        """Create a new translation job in the database.
        
        Args:
            book_id: ID of the book
            total_pages: Total number of pages to translate
            ollama_model: Ollama model to use
            ollama_base_url: Ollama base URL
            use_refinement: Whether to use two-step refinement
            temperature: Temperature setting
            
        Returns:
            ID of the newly created job
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO translation_jobs 
                (book_id, status, total_pages, ollama_model, ollama_base_url, use_refinement, temperature, parallel_workers, started_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                book_id,
                'pending',
                total_pages,
                ollama_model,
                ollama_base_url,
                use_refinement,
                temperature,
                parallel_workers,
                datetime.now()
            ))
            
            job_id = cursor.lastrowid
            conn.commit()
            return job_id
        finally:
            conn.close()
    
    def get_active_job_for_book(self, book_id: int) -> Optional[Dict]:
        """Get the active (pending or running) translation job for a book.
        
        Args:
            book_id: ID of the book
            
        Returns:
            Dictionary containing job information, or None if no active job found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM translation_jobs 
                WHERE book_id = ? AND status IN ('pending', 'running')
                ORDER BY started_at DESC
                LIMIT 1
            """, (book_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
    
    def update_job_progress(
        self,
        job_id: int,
        completed_pages: Optional[int] = None,
        failed_pages: Optional[int] = None,
        avg_time_per_page: Optional[float] = None,
        estimated_time_remaining: Optional[float] = None,
        status: Optional[str] = None
    ) -> bool:
        """Update progress for a translation job.
        
        Args:
            job_id: ID of the job
            completed_pages: Number of completed pages
            failed_pages: Number of failed pages
            avg_time_per_page: Average time per page in seconds
            estimated_time_remaining: Estimated time remaining in seconds
            status: Job status (pending, running, completed, failed)
            
        Returns:
            True if update was successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            updates = []
            params = []
            
            if completed_pages is not None:
                updates.append("completed_pages = ?")
                params.append(completed_pages)
            
            if failed_pages is not None:
                updates.append("failed_pages = ?")
                params.append(failed_pages)
            
            if avg_time_per_page is not None:
                updates.append("avg_time_per_page = ?")
                params.append(avg_time_per_page)
            
            if estimated_time_remaining is not None:
                updates.append("estimated_time_remaining = ?")
                params.append(estimated_time_remaining)
            
            if status is not None:
                updates.append("status = ?")
                params.append(status)
            
            if not updates:
                return False
            
            params.append(job_id)
            
            cursor.execute(f"""
                UPDATE translation_jobs 
                SET {', '.join(updates)} 
                WHERE id = ?
            """, params)
            
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def complete_job(self, job_id: int, status: str, error_message: Optional[str] = None) -> bool:
        """Mark a translation job as completed or failed.
        
        Args:
            job_id: ID of the job
            status: Final status ('completed' or 'failed')
            error_message: Error message if status is 'failed'
            
        Returns:
            True if update was successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE translation_jobs 
                SET status = ?, completed_at = ?, error_message = ?
                WHERE id = ?
            """, (status, datetime.now(), error_message, job_id))
            
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def get_job(self, job_id: int) -> Optional[Dict]:
        """Get a translation job by ID.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Dictionary containing job information, or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM translation_jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
    
    def get_translation_job(self, job_id: int) -> Optional[Dict]:
        """Get a translation job by ID (alias for get_job).
        
        Args:
            job_id: ID of the job
            
        Returns:
            Dictionary containing job information, or None if not found
        """
        return self.get_job(job_id)

