"""Term Database Manager - SQLite database for managing terms and their content.

This module provides a database manager for storing:
- Terms (topics/domains being researched)
- Term content (fetched content for each term)
- Term links (URLs visited and not visited)
- Metadata (relevance scores, confidence, timestamps, etc.)
"""

import sqlite3
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class TermDBManager:
    """Manages SQLite database for terms, content, and links."""
    
    def __init__(self, db_path: str = None):
        """Initialize the Term Database Manager.
        
        Args:
            db_path: Path to SQLite database file (default: .term_db/terms.db)
        """
        if db_path is None:
            # Use project root directory
            project_root = Path(__file__).parent.parent.parent
            db_dir = project_root / ".term_db"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "terms.db")
        
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create terms table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS terms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT NOT NULL UNIQUE,
                original_url TEXT,
                domain TEXT,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_gathered_at TIMESTAMP,
                content_count INTEGER DEFAULT 0,
                links_count INTEGER DEFAULT 0,
                unvisited_links_count INTEGER DEFAULT 0
            )
        """)
        
        # Create term_content table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS term_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                source_url TEXT,
                relevance_score REAL,
                confidence_score REAL,
                chunk_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (term_id) REFERENCES terms(id) ON DELETE CASCADE
            )
        """)
        
        # Create term_links table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS term_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term_id INTEGER NOT NULL,
                url TEXT NOT NULL,
                domain TEXT,
                visited BOOLEAN DEFAULT 0,
                visited_at TIMESTAMP,
                relevance_score REAL,
                is_relevant BOOLEAN DEFAULT 0,
                link_type TEXT DEFAULT 'external',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (term_id) REFERENCES terms(id) ON DELETE CASCADE,
                UNIQUE(term_id, url)
            )
        """)
        
        # Create domain_blacklist table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS domain_blacklist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term_id INTEGER NOT NULL,
                domain TEXT NOT NULL,
                pages_visited INTEGER DEFAULT 0,
                pages_relevant INTEGER DEFAULT 0,
                relevance_ratio REAL DEFAULT 0.0,
                blacklisted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reason TEXT,
                FOREIGN KEY (term_id) REFERENCES terms(id) ON DELETE CASCADE,
                UNIQUE(term_id, domain)
            )
        """)
        
        # Migrate existing data: add domain column to term_links if it doesn't exist
        # This must happen BEFORE creating indexes
        try:
            cursor.execute("ALTER TABLE term_links ADD COLUMN domain TEXT")
        except sqlite3.OperationalError:
            # Column already exists, ignore
            pass
        
        try:
            cursor.execute("ALTER TABLE term_links ADD COLUMN is_relevant BOOLEAN DEFAULT 0")
        except sqlite3.OperationalError:
            # Column already exists, ignore
            pass
        
        # Create indexes for better query performance (after columns are ensured to exist)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_terms_term ON terms(term)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_terms_status ON terms(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_term_content_term_id ON term_content(term_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_term_links_term_id ON term_links(term_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_term_links_visited ON term_links(visited)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_term_links_url ON term_links(url)")
        
        # Only create domain index if domain column exists
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_term_links_domain ON term_links(domain)")
        except sqlite3.OperationalError:
            # Column might not exist, skip index creation
            logger.warning("Could not create domain index - column may not exist")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_domain_blacklist_term_id ON domain_blacklist(term_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_domain_blacklist_domain ON domain_blacklist(domain)")
        
        conn.commit()
        conn.close()
        logger.info(f"Initialized term database at: {self.db_path}")
    
    def _get_domain_from_url(self, url: str) -> Optional[str]:
        """Extract domain from URL.
        
        Args:
            url: URL string
            
        Returns:
            Domain name or None
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove 'www.' if present
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain if domain else None
        except Exception as e:
            logger.warning(f"Error extracting domain from URL {url}: {str(e)}")
            return None
    
    def add_term(
        self,
        term: str,
        original_url: str = None,
        domain: str = None,
        status: str = 'active'
    ) -> int:
        """Add a new term to the database.
        
        Args:
            term: Term/topic name
            original_url: Original URL that led to this term
            domain: Domain extracted from URL
            status: Status (active, completed, archived)
            
        Returns:
            Term ID (int)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Try to get existing term
            cursor.execute("SELECT id FROM terms WHERE term = ?", (term,))
            existing = cursor.fetchone()
            
            if existing:
                term_id = existing[0]
                # Update existing term
                cursor.execute("""
                    UPDATE terms 
                    SET updated_at = CURRENT_TIMESTAMP,
                        original_url = COALESCE(?, original_url),
                        domain = COALESCE(?, domain),
                        status = ?
                    WHERE id = ?
                """, (original_url, domain, status, term_id))
                logger.info(f"Updated existing term: {term} (ID: {term_id})")
            else:
                # Insert new term
                cursor.execute("""
                    INSERT INTO terms (term, original_url, domain, status)
                    VALUES (?, ?, ?, ?)
                """, (term, original_url, domain, status))
                term_id = cursor.lastrowid
                logger.info(f"Added new term: {term} (ID: {term_id})")
            
            conn.commit()
            return term_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Error adding term: {str(e)}")
            raise
        finally:
            conn.close()
    
    def add_content(
        self,
        term_id: int,
        content: str,
        source_url: str = None,
        relevance_score: float = None,
        confidence_score: float = None,
        chunk_count: int = 0
    ) -> int:
        """Add content for a term.
        
        Args:
            term_id: Term ID
            content: Content text
            source_url: Source URL where content was fetched
            relevance_score: Relevance score (0.0 to 1.0)
            confidence_score: Confidence score (0.0 to 1.0)
            chunk_count: Number of chunks this content was split into
            
        Returns:
            Content ID (int)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO term_content 
                (term_id, content, source_url, relevance_score, confidence_score, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (term_id, content, source_url, relevance_score, confidence_score, chunk_count))
            
            content_id = cursor.lastrowid
            
            # Update term's content count
            cursor.execute("""
                UPDATE terms 
                SET content_count = content_count + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (term_id,))
            
            conn.commit()
            logger.info(f"Added content for term ID {term_id} (Content ID: {content_id})")
            return content_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Error adding content: {str(e)}")
            raise
        finally:
            conn.close()
    
    def add_links(
        self,
        term_id: int,
        urls: List[str],
        visited: bool = False,
        link_type: str = 'external'
    ) -> int:
        """Add links for a term.
        
        Args:
            term_id: Term ID
            urls: List of URLs
            visited: Whether links have been visited
            link_type: Type of link (external, internal, related)
            
        Returns:
            Number of links added
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            added_count = 0
            for url in urls:
                try:
                    # Extract domain from URL
                    domain = self._get_domain_from_url(url)
                    
                    cursor.execute("""
                        INSERT OR IGNORE INTO term_links 
                        (term_id, url, domain, visited, link_type)
                        VALUES (?, ?, ?, ?, ?)
                    """, (term_id, url, domain, 1 if visited else 0, link_type))
                    
                    if cursor.rowcount > 0:
                        added_count += 1
                except Exception as e:
                    logger.warning(f"Error adding link {url}: {str(e)}")
                    continue
            
            # Update term's link counts
            if added_count > 0:
                cursor.execute("""
                    UPDATE terms 
                    SET links_count = (SELECT COUNT(*) FROM term_links WHERE term_id = ?),
                        unvisited_links_count = (SELECT COUNT(*) FROM term_links WHERE term_id = ? AND visited = 0),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (term_id, term_id, term_id))
            
            conn.commit()
            logger.info(f"Added {added_count} links for term ID {term_id}")
            return added_count
        except Exception as e:
            conn.rollback()
            logger.error(f"Error adding links: {str(e)}")
            raise
        finally:
            conn.close()
    
    def mark_link_visited(self, term_id: int, url: str):
        """Mark a link as visited.
        
        Args:
            term_id: Term ID
            url: URL to mark as visited
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Extract domain if not already set
            domain = self._get_domain_from_url(url)
            
            if domain:
                cursor.execute("""
                    UPDATE term_links 
                    SET visited = 1, visited_at = CURRENT_TIMESTAMP, domain = ?
                    WHERE term_id = ? AND url = ?
                """, (domain, term_id, url))
            else:
                cursor.execute("""
                    UPDATE term_links 
                    SET visited = 1, visited_at = CURRENT_TIMESTAMP
                    WHERE term_id = ? AND url = ?
                """, (term_id, url))
            
            # Update unvisited count
            cursor.execute("""
                UPDATE terms 
                SET unvisited_links_count = (SELECT COUNT(*) FROM term_links WHERE term_id = ? AND visited = 0),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (term_id, term_id))
            
            conn.commit()
            logger.info(f"Marked link as visited: {url} for term ID {term_id}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error marking link as visited: {str(e)}")
            raise
        finally:
            conn.close()
    
    def get_term(self, term_id: int = None, term_name: str = None) -> Optional[Dict]:
        """Get a term by ID or name.
        
        Args:
            term_id: Term ID
            term_name: Term name
            
        Returns:
            Dictionary with term information or None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            if term_id:
                cursor.execute("SELECT * FROM terms WHERE id = ?", (term_id,))
            elif term_name:
                cursor.execute("SELECT * FROM terms WHERE term = ?", (term_name,))
            else:
                return None
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()
    
    def list_terms(self, status: str = None) -> List[Dict]:
        """List all terms.
        
        Args:
            status: Filter by status (active, completed, archived)
            
        Returns:
            List of dictionaries with term information
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            if status:
                cursor.execute("SELECT * FROM terms WHERE status = ? ORDER BY updated_at DESC", (status,))
            else:
                cursor.execute("SELECT * FROM terms ORDER BY updated_at DESC")
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def get_term_content(self, term_id: int) -> List[Dict]:
        """Get all content for a term.
        
        Args:
            term_id: Term ID
            
        Returns:
            List of dictionaries with content information
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM term_content 
                WHERE term_id = ? 
                ORDER BY created_at DESC
            """, (term_id,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def delete_content(self, term_id: int, content_id: int) -> bool:
        """Delete a content item by ID.
        
        Args:
            term_id: Term ID
            content_id: Content ID to delete
            
        Returns:
            True if content was deleted, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Verify the content belongs to this term
            cursor.execute("""
                SELECT id FROM term_content 
                WHERE id = ? AND term_id = ?
            """, (content_id, term_id))
            
            if not cursor.fetchone():
                return False
            
            # Delete the content
            cursor.execute("""
                DELETE FROM term_content 
                WHERE id = ? AND term_id = ?
            """, (content_id, term_id))
            
            deleted = cursor.rowcount > 0
            
            # Update content count
            if deleted:
                cursor.execute("""
                    UPDATE terms 
                    SET content_count = (SELECT COUNT(*) FROM term_content WHERE term_id = ?),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (term_id, term_id))
            
            conn.commit()
            if deleted:
                logger.info(f"Deleted content ID {content_id} for term ID {term_id}")
            return deleted
        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting content: {str(e)}")
            raise
        finally:
            conn.close()
    
    def get_term_links(
        self,
        term_id: int,
        visited: bool = None,
        limit: int = None
    ) -> List[Dict]:
        """Get links for a term.
        
        Args:
            term_id: Term ID
            visited: Filter by visited status (True/False/None for all)
            limit: Maximum number of links to return
            
        Returns:
            List of dictionaries with link information
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            query = "SELECT * FROM term_links WHERE term_id = ?"
            params = [term_id]
            
            if visited is not None:
                query += " AND visited = ?"
                params.append(1 if visited else 0)
            
            query += " ORDER BY created_at DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def get_unvisited_links(self, term_id: int, limit: int = None) -> List[Dict]:
        """Get unvisited links for a term, excluding blacklisted domains.
        
        Args:
            term_id: Term ID
            limit: Maximum number of links to return
            
        Returns:
            List of dictionaries with unvisited link information (excluding blacklisted domains)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # Get blacklisted domains for this term
            cursor.execute("""
                SELECT domain FROM domain_blacklist WHERE term_id = ?
            """, (term_id,))
            blacklisted_domains = [row[0] for row in cursor.fetchall()]
            
            # Get unvisited links, excluding blacklisted domains
            query = """
                SELECT * FROM term_links 
                WHERE term_id = ? AND visited = 0
            """
            params = [term_id]
            
            if blacklisted_domains:
                placeholders = ','.join(['?'] * len(blacklisted_domains))
                query += f" AND (domain IS NULL OR domain NOT IN ({placeholders}))"
                params.extend(blacklisted_domains)
            
            query += " ORDER BY created_at DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def update_link_relevance(self, term_id: int, url: str, is_relevant: bool, relevance_score: float = None):
        """Update relevance information for a link.
        
        Args:
            term_id: Term ID
            url: URL
            is_relevant: Whether the link is relevant to the term
            relevance_score: Optional relevance score (0.0 to 1.0)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if relevance_score is not None:
                cursor.execute("""
                    UPDATE term_links 
                    SET is_relevant = ?, relevance_score = ?
                    WHERE term_id = ? AND url = ?
                """, (1 if is_relevant else 0, relevance_score, term_id, url))
            else:
                cursor.execute("""
                    UPDATE term_links 
                    SET is_relevant = ?
                    WHERE term_id = ? AND url = ?
                """, (1 if is_relevant else 0, term_id, url))
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating link relevance: {str(e)}")
            raise
        finally:
            conn.close()
    
    def get_domain_statistics(self, term_id: int, relevance_threshold: float = 0.2) -> Dict[str, Dict]:
        """Get statistics for each domain.
        
        Args:
            term_id: Term ID
            relevance_threshold: Threshold for counting low-relevance links (default: 0.2)
            
        Returns:
            Dictionary mapping domain to statistics:
                - pages_visited: int
                - pages_relevant: int
                - relevance_ratio: float (0.0 to 1.0)
                - low_relevance_count: int (count of links with relevance < threshold)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    domain,
                    COUNT(*) as pages_visited,
                    SUM(CASE WHEN is_relevant = 1 THEN 1 ELSE 0 END) as pages_relevant,
                    SUM(CASE WHEN relevance_score IS NOT NULL AND relevance_score < ? THEN 1 ELSE 0 END) as low_relevance_count
                FROM term_links
                WHERE term_id = ? AND visited = 1 AND domain IS NOT NULL
                GROUP BY domain
            """, (relevance_threshold, term_id))
            
            stats = {}
            for row in cursor.fetchall():
                domain, pages_visited, pages_relevant, low_relevance_count = row
                relevance_ratio = pages_relevant / pages_visited if pages_visited > 0 else 0.0
                stats[domain] = {
                    'pages_visited': pages_visited,
                    'pages_relevant': pages_relevant,
                    'relevance_ratio': relevance_ratio,
                    'low_relevance_count': low_relevance_count or 0  # Count of links with relevance < 0.2
                }
                
            return stats
        finally:
            conn.close()
    
    def blacklist_domain(self, term_id: int, domain: str, pages_visited: int, pages_relevant: int, reason: str = None):
        """Blacklist a domain for a term.
        
        Args:
            term_id: Term ID
            domain: Domain to blacklist
            pages_visited: Number of pages visited from this domain
            pages_relevant: Number of relevant pages from this domain
            reason: Optional reason for blacklisting
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            relevance_ratio = pages_relevant / pages_visited if pages_visited > 0 else 0.0
            
            cursor.execute("""
                INSERT OR REPLACE INTO domain_blacklist 
                (term_id, domain, pages_visited, pages_relevant, relevance_ratio, reason, blacklisted_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (term_id, domain, pages_visited, pages_relevant, relevance_ratio, reason))
            
            conn.commit()
            logger.info(f"Blacklisted domain {domain} for term ID {term_id} (relevance: {relevance_ratio:.2%})")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error blacklisting domain: {str(e)}")
            raise
        finally:
            conn.close()
    
    def is_domain_blacklisted(self, term_id: int, domain: str) -> bool:
        """Check if a domain is blacklisted for a term.
        
        Args:
            term_id: Term ID
            domain: Domain to check
            
        Returns:
            True if blacklisted, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT COUNT(*) FROM domain_blacklist 
                WHERE term_id = ? AND domain = ?
            """, (term_id, domain))
            
            return cursor.fetchone()[0] > 0
        finally:
            conn.close()
    
    def get_blacklisted_domains(self, term_id: int) -> List[Dict]:
        """Get all blacklisted domains for a term.
        
        Args:
            term_id: Term ID
            
        Returns:
            List of dictionaries with blacklist information
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM domain_blacklist 
                WHERE term_id = ?
                ORDER BY blacklisted_at DESC
            """, (term_id,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def delete_link(self, term_id: int, url: str) -> bool:
        """Delete a single link by URL.
        
        Args:
            term_id: Term ID
            url: URL to delete
            
        Returns:
            True if link was deleted, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                DELETE FROM term_links 
                WHERE term_id = ? AND url = ?
            """, (term_id, url))
            
            deleted = cursor.rowcount > 0
            
            # Update link counts
            if deleted:
                cursor.execute("""
                    UPDATE terms 
                    SET unvisited_links_count = (SELECT COUNT(*) FROM term_links WHERE term_id = ? AND visited = 0),
                        links_count = (SELECT COUNT(*) FROM term_links WHERE term_id = ?),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (term_id, term_id, term_id))
            
            conn.commit()
            if deleted:
                logger.info(f"Deleted link: {url} for term ID {term_id}")
            return deleted
        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting link: {str(e)}")
            raise
        finally:
            conn.close()
    
    def delete_unvisited_links_by_domains(self, term_id: int, domains: List[str]) -> int:
        """Delete unvisited links from specific domains.
        
        Args:
            term_id: Term ID
            domains: List of domains to delete links from
            
        Returns:
            Number of links deleted
        """
        if not domains:
            return 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            placeholders = ','.join(['?'] * len(domains))
            cursor.execute(f"""
                DELETE FROM term_links 
                WHERE term_id = ? AND visited = 0 AND domain IN ({placeholders})
            """, [term_id] + domains)
            
            deleted_count = cursor.rowcount
            
            # Update unvisited count
            cursor.execute("""
                UPDATE terms 
                SET unvisited_links_count = (SELECT COUNT(*) FROM term_links WHERE term_id = ? AND visited = 0),
                    links_count = (SELECT COUNT(*) FROM term_links WHERE term_id = ?),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (term_id, term_id, term_id))
            
            conn.commit()
            logger.info(f"Deleted {deleted_count} unvisited links from {len(domains)} domain(s) for term ID {term_id}")
            return deleted_count
        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting links by domains: {str(e)}")
            raise
        finally:
            conn.close()
    
    def update_term_last_gathered(self, term_id: int):
        """Update the last_gathered_at timestamp for a term.
        
        Args:
            term_id: Term ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE terms 
                SET last_gathered_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (term_id,))
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating last_gathered_at: {str(e)}")
            raise
        finally:
            conn.close()
    
    def update_term_status(self, term_id: int, status: str):
        """Update term status.
        
        Args:
            term_id: Term ID
            status: New status (active, completed, archived)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE terms 
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (status, term_id))
            
            conn.commit()
            logger.info(f"Updated term ID {term_id} status to {status}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating term status: {str(e)}")
            raise
        finally:
            conn.close()
    
    def delete_term(self, term_id: int):
        """Delete a term and all its related data (cascade).
        
        Args:
            term_id: Term ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM terms WHERE id = ?", (term_id,))
            conn.commit()
            logger.info(f"Deleted term ID {term_id} and all related data")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting term: {str(e)}")
            raise
        finally:
            conn.close()
    
    def get_term_stats(self, term_id: int) -> Dict:
        """Get statistics for a term.
        
        Args:
            term_id: Term ID
            
        Returns:
            Dictionary with statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            stats = {}
            
            # Get content count and total chunks
            cursor.execute("""
                SELECT COUNT(*), SUM(chunk_count) 
                FROM term_content 
                WHERE term_id = ?
            """, (term_id,))
            row = cursor.fetchone()
            stats['content_count'] = row[0] or 0
            stats['total_chunks'] = row[1] or 0
            
            # Get link counts
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN visited = 1 THEN 1 ELSE 0 END) as visited,
                    SUM(CASE WHEN visited = 0 THEN 1 ELSE 0 END) as unvisited
                FROM term_links 
                WHERE term_id = ?
            """, (term_id,))
            row = cursor.fetchone()
            stats['total_links'] = row[0] or 0
            stats['visited_links'] = row[1] or 0
            stats['unvisited_links'] = row[2] or 0
            
            # Get average relevance score
            cursor.execute("""
                SELECT AVG(relevance_score) 
                FROM term_content 
                WHERE term_id = ? AND relevance_score IS NOT NULL
            """, (term_id,))
            row = cursor.fetchone()
            stats['avg_relevance_score'] = float(row[0]) if row[0] is not None else 0.0
            
            return stats
        finally:
            conn.close()

