"""SQLite database management for RSS feed tracking."""

import sqlite3
import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta

class RSSDB:
    """Manages SQLite database for RSS feed tracking."""
    
    def __init__(self, db_path: str = None):
        """Initialize the database connection.
        
        Args:
            db_path: Path to SQLite database file (default: rss_feeds.db in .chroma_db/rss_feeds/)
        """
        if db_path is None:
            db_dir = os.path.join(".chroma_db", "rss_feeds")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "rss_feeds.db")
        
        self.db_path = db_path
        self._initialize_database()
    
    def _get_connection(self):
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _initialize_database(self):
        """Initialize database tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # RSS Feeds table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rss_feeds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                check_frequency_per_day INTEGER NOT NULL CHECK(check_frequency_per_day > 0),
                last_checked TIMESTAMP,
                next_check TIMESTAMP,
                is_active INTEGER DEFAULT 1 CHECK(is_active IN (0, 1)),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Articles/Links table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feed_id INTEGER NOT NULL,
                title TEXT,
                link TEXT NOT NULL,
                description TEXT,
                published_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (feed_id) REFERENCES rss_feeds(id) ON DELETE CASCADE,
                UNIQUE(feed_id, link)
            )
        """)
        
        # Domain pause table (tracks paused domains)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS domain_pauses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL UNIQUE,
                is_paused INTEGER DEFAULT 1 CHECK(is_paused IN (0, 1)),
                pause_reason TEXT,
                paused_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resumed_at TIMESTAMP,
                error_count INTEGER DEFAULT 1,
                last_error TEXT,
                last_error_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Keywords table (for article topic matching)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL UNIQUE,
                description TEXT,
                is_active INTEGER DEFAULT 1 CHECK(is_active IN (0, 1)),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_articles_feed_id ON articles(feed_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_articles_created_at ON articles(created_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rss_feeds_next_check ON rss_feeds(next_check)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rss_feeds_is_active ON rss_feeds(is_active)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_domain_pauses_domain ON domain_pauses(domain)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_domain_pauses_is_paused ON domain_pauses(is_paused)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_keywords_keyword ON keywords(keyword)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_keywords_is_active ON keywords(is_active)
        """)
        
        conn.commit()
        conn.close()
    
    # RSS Feed management
    def add_feed(self, name: str, url: str, check_frequency_per_day: int) -> tuple[bool, str]:
        """Add a new RSS feed.
        
        Args:
            name: Name of the feed
            url: RSS feed URL
            check_frequency_per_day: Number of times per day to check for updates
            
        Returns:
            Tuple of (success, message)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            # Calculate next check time (now + interval based on frequency)
            now = datetime.now()
            hours_between_checks = 24 / check_frequency_per_day
            next_check = now + timedelta(hours=hours_between_checks)
            
            cursor.execute("""
                INSERT INTO rss_feeds (name, url, check_frequency_per_day, last_checked, next_check, is_active)
                VALUES (?, ?, ?, ?, ?, 1)
            """, (name, url, check_frequency_per_day, now, next_check))
            
            feed_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return True, f"✅ Feed '{name}' added (ID: {feed_id})"
        except sqlite3.IntegrityError:
            conn.close()
            return False, f"❌ Feed with URL '{url}' already exists"
        except Exception as e:
            conn.close()
            return False, f"❌ Error adding feed: {str(e)}"
    
    def get_feeds(self, active_only: bool = False) -> List[Dict]:
        """Get all RSS feeds.
        
        Args:
            active_only: If True, only return active feeds
            
        Returns:
            List of feed dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if active_only:
            cursor.execute("""
                SELECT * FROM rss_feeds 
                WHERE is_active = 1 
                ORDER BY name
            """)
        else:
            cursor.execute("""
                SELECT * FROM rss_feeds 
                ORDER BY name
            """)
        
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_feed(self, feed_id: int) -> Optional[Dict]:
        """Get a single feed by ID.
        
        Args:
            feed_id: Feed ID
            
        Returns:
            Feed dictionary or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rss_feeds WHERE id = ?", (feed_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def update_feed(self, feed_id: int, name: str = None, url: str = None, 
                   check_frequency_per_day: int = None, is_active: int = None) -> tuple[bool, str]:
        """Update an RSS feed.
        
        Args:
            feed_id: Feed ID
            name: New name (optional)
            url: New URL (optional)
            check_frequency_per_day: New check frequency (optional)
            is_active: Active status (0 or 1, optional)
            
        Returns:
            Tuple of (success, message)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            updates = []
            params = []
            
            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if url is not None:
                updates.append("url = ?")
                params.append(url)
            if check_frequency_per_day is not None:
                updates.append("check_frequency_per_day = ?")
                params.append(check_frequency_per_day)
                # Recalculate next_check if frequency changed
                now = datetime.now()
                hours_between_checks = 24 / check_frequency_per_day
                updates.append("next_check = ?")
                params.append(now + timedelta(hours=hours_between_checks))
            if is_active is not None:
                updates.append("is_active = ?")
                params.append(is_active)
            
            if not updates:
                conn.close()
                return False, "❌ No fields to update"
            
            updates.append("updated_at = ?")
            params.append(datetime.now())
            params.append(feed_id)
            
            query = f"UPDATE rss_feeds SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)
            conn.commit()
            conn.close()
            return True, "✅ Feed updated"
        except sqlite3.IntegrityError:
            conn.close()
            return False, "❌ Feed with this URL already exists"
        except Exception as e:
            conn.close()
            return False, f"❌ Error updating feed: {str(e)}"
    
    def delete_feed(self, feed_id: int) -> tuple[bool, str]:
        """Delete an RSS feed and all its articles.
        
        Args:
            feed_id: Feed ID
            
        Returns:
            Tuple of (success, message)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            # Delete feed (cascade will delete articles)
            cursor.execute("DELETE FROM rss_feeds WHERE id = ?", (feed_id,))
            conn.commit()
            conn.close()
            return True, "✅ Feed deleted"
        except Exception as e:
            conn.close()
            return False, f"❌ Error deleting feed: {str(e)}"
    
    def update_last_checked(self, feed_id: int) -> None:
        """Update the last_checked and next_check timestamps for a feed.
        
        Args:
            feed_id: Feed ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get feed to calculate next check
        cursor.execute("SELECT check_frequency_per_day FROM rss_feeds WHERE id = ?", (feed_id,))
        row = cursor.fetchone()
        if row:
            check_frequency = row['check_frequency_per_day']
            now = datetime.now()
            hours_between_checks = 24 / check_frequency
            next_check = now + timedelta(hours=hours_between_checks)
            
            cursor.execute("""
                UPDATE rss_feeds 
                SET last_checked = ?, next_check = ?, updated_at = ?
                WHERE id = ?
            """, (now, next_check, now, feed_id))
            conn.commit()
        
        conn.close()
    
    def get_feeds_due_for_check(self, limit: int = 10) -> List[Dict]:
        """Get feeds that are due for checking.
        
        Args:
            limit: Maximum number of feeds to return
            
        Returns:
            List of feed dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now()
        
        cursor.execute("""
            SELECT * FROM rss_feeds 
            WHERE is_active = 1 
            AND (next_check IS NULL OR next_check <= ?)
            ORDER BY next_check ASC
            LIMIT ?
        """, (now, limit))
        
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    # Article management
    def add_article(self, feed_id: int, title: str, link: str, 
                   description: str = None, published_at: datetime = None) -> tuple[bool, str]:
        """Add a new article. Only adds if article doesn't already exist.
        
        Args:
            feed_id: Feed ID
            title: Article title
            link: Article URL
            description: Article description (optional)
            published_at: Publication date (optional)
            
        Returns:
            Tuple of (success, message)
            - success: True if article was added, False if it already exists or error occurred
            - message: Success or error message
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO articles (feed_id, title, link, description, published_at)
                VALUES (?, ?, ?, ?, ?)
            """, (feed_id, title, link, description, published_at))
            
            article_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return True, f"✅ Article added (ID: {article_id})"
        except sqlite3.IntegrityError:
            # Article already exists (UNIQUE constraint on feed_id, link)
            conn.close()
            return False, "Article already exists"
        except Exception as e:
            conn.close()
            return False, f"❌ Error adding article: {str(e)}"
    
    def get_articles(self, feed_id: int = None, limit: int = None, 
                    order_by: str = "published_at") -> List[Dict]:
        """Get articles.
        
        Args:
            feed_id: Filter by feed ID (optional)
            limit: Maximum number of articles to return (optional)
            order_by: Field to order by (default: published_at)
            
        Returns:
            List of article dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT a.*, f.name as feed_name, f.url as feed_url
            FROM articles a
            JOIN rss_feeds f ON a.feed_id = f.id
            WHERE 1=1
        """
        params = []
        
        if feed_id:
            query += " AND a.feed_id = ?"
            params.append(feed_id)
        
        # Validate order_by to prevent SQL injection
        allowed_order_by = ["published_at", "created_at", "title"]
        if order_by not in allowed_order_by:
            order_by = "published_at"
        
        query += f" ORDER BY a.{order_by} DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_article_count(self, feed_id: int = None) -> int:
        """Get count of articles.
        
        Args:
            feed_id: Filter by feed ID (optional)
            
        Returns:
            Number of articles
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if feed_id:
            cursor.execute("SELECT COUNT(*) as count FROM articles WHERE feed_id = ?", (feed_id,))
        else:
            cursor.execute("SELECT COUNT(*) as count FROM articles")
        
        row = cursor.fetchone()
        conn.close()
        return row['count'] if row else 0
    
    def delete_article(self, article_id: int) -> tuple[bool, str]:
        """Delete an article.
        
        Args:
            article_id: Article ID
            
        Returns:
            Tuple of (success, message)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM articles WHERE id = ?", (article_id,))
            conn.commit()
            conn.close()
            return True, "✅ Article deleted"
        except Exception as e:
            conn.close()
            return False, f"❌ Error deleting article: {str(e)}"
    
    def delete_articles_by_feed(self, feed_id: int) -> tuple[bool, str]:
        """Delete all articles for a feed.
        
        Args:
            feed_id: Feed ID
            
        Returns:
            Tuple of (success, message)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM articles WHERE feed_id = ?", (feed_id,))
            conn.commit()
            conn.close()
            return True, "✅ Articles deleted"
        except Exception as e:
            conn.close()
            return False, f"❌ Error deleting articles: {str(e)}"
    
    def delete_old_articles(self, days_old: int = 10) -> tuple[bool, str, int]:
        """Delete article links older than specified days based on published date.
        
        Args:
            days_old: Number of days old (default: 10)
            
        Returns:
            Tuple of (success, message, count of deleted articles)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            # Calculate cutoff date
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Count articles to be deleted (use published_at, fallback to created_at if published_at is NULL)
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM articles 
                WHERE COALESCE(published_at, created_at) < ?
            """, (cutoff_date,))
            count_row = cursor.fetchone()
            count = count_row['count'] if count_row else 0
            
            if count == 0:
                conn.close()
                return True, "No old article links to delete", 0
            
            # Delete old article links (not feed links - only articles)
            cursor.execute("""
                DELETE FROM articles 
                WHERE COALESCE(published_at, created_at) < ?
            """, (cutoff_date,))
            
            conn.commit()
            conn.close()
            return True, f"✅ Deleted {count} article link(s) older than {days_old} days", count
        except Exception as e:
            conn.close()
            return False, f"❌ Error deleting old article links: {str(e)}", 0
    
    def get_old_articles_count(self, days_old: int = 10) -> int:
        """Get count of article links older than specified days based on published date.
        
        Args:
            days_old: Number of days old (default: 10)
            
        Returns:
            Count of old article links
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Use published_at if available, otherwise use created_at
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM articles 
                WHERE COALESCE(published_at, created_at) < ?
            """, (cutoff_date,))
            
            row = cursor.fetchone()
            conn.close()
            return row['count'] if row else 0
        except Exception as e:
            conn.close()
            return 0
    
    # Domain pause management
    def is_domain_paused(self, domain: str) -> bool:
        """Check if a domain is paused.
        
        Args:
            domain: Domain name
            
        Returns:
            True if domain is paused, False otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT is_paused FROM domain_pauses 
                WHERE domain = ? AND is_paused = 1
            """, (domain.lower(),))
            row = cursor.fetchone()
            conn.close()
            return row is not None
        except Exception as e:
            conn.close()
            return False
    
    def pause_domain(self, domain: str, reason: str = None, error_message: str = None) -> tuple[bool, str]:
        """Pause a domain due to connection errors.
        
        Args:
            domain: Domain name to pause
            reason: Reason for pausing (optional)
            error_message: Last error message (optional)
            
        Returns:
            Tuple of (success, message)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            domain = domain.lower()
            now = datetime.now()
            
            # Check if domain already exists
            cursor.execute("SELECT id, error_count FROM domain_pauses WHERE domain = ?", (domain,))
            row = cursor.fetchone()
            
            if row:
                # Update existing pause
                error_count = row['error_count'] + 1
                cursor.execute("""
                    UPDATE domain_pauses 
                    SET is_paused = 1,
                        pause_reason = COALESCE(?, pause_reason),
                        paused_at = ?,
                        error_count = ?,
                        last_error = ?,
                        last_error_at = ?
                    WHERE domain = ?
                """, (reason, now, error_count, error_message, now, domain))
            else:
                # Insert new pause
                cursor.execute("""
                    INSERT INTO domain_pauses (domain, is_paused, pause_reason, paused_at, error_count, last_error, last_error_at)
                    VALUES (?, 1, ?, ?, 1, ?, ?)
                """, (domain, reason, now, error_message, now))
            
            conn.commit()
            conn.close()
            return True, f"Domain {domain} paused"
        except Exception as e:
            conn.close()
            return False, f"Error pausing domain: {str(e)}"
    
    def resume_domain(self, domain: str) -> tuple[bool, str]:
        """Resume a paused domain.
        
        Args:
            domain: Domain name to resume
            
        Returns:
            Tuple of (success, message)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            domain = domain.lower()
            now = datetime.now()
            
            cursor.execute("""
                UPDATE domain_pauses 
                SET is_paused = 0,
                    resumed_at = ?
                WHERE domain = ?
            """, (now, domain))
            
            conn.commit()
            conn.close()
            
            if cursor.rowcount > 0:
                return True, f"Domain {domain} resumed"
            else:
                return False, f"Domain {domain} not found or not paused"
        except Exception as e:
            conn.close()
            return False, f"Error resuming domain: {str(e)}"
    
    def get_paused_domains(self) -> List[Dict]:
        """Get all paused domains.
        
        Returns:
            List of paused domain dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT * FROM domain_pauses 
                WHERE is_paused = 1 
                ORDER BY paused_at DESC
            """)
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:
            conn.close()
            return []
    
    def get_domain_pause_info(self, domain: str) -> Optional[Dict]:
        """Get pause information for a domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Domain pause dictionary or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM domain_pauses WHERE domain = ?", (domain.lower(),))
            row = cursor.fetchone()
            conn.close()
            return dict(row) if row else None
        except Exception as e:
            conn.close()
            return None
    
    # Keyword management
    def add_keyword(self, keyword: str, description: str = None) -> tuple[bool, str]:
        """Add a new keyword.
        
        Args:
            keyword: Keyword to track
            description: Optional description of the keyword
            
        Returns:
            Tuple of (success, message)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO keywords (keyword, description, is_active)
                VALUES (?, ?, 1)
            """, (keyword.strip(), description.strip() if description and description.strip() else None))
            
            keyword_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return True, f"✅ Keyword added (ID: {keyword_id})"
        except sqlite3.IntegrityError:
            conn.close()
            return False, "❌ Keyword already exists"
        except Exception as e:
            conn.close()
            return False, f"❌ Error adding keyword: {str(e)}"
    
    def get_keywords(self, active_only: bool = False) -> List[Dict]:
        """Get all keywords.
        
        Args:
            active_only: If True, return only active keywords
            
        Returns:
            List of keyword dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if active_only:
            cursor.execute("""
                SELECT id, keyword, description, is_active, created_at, updated_at
                FROM keywords
                WHERE is_active = 1
                ORDER BY keyword ASC
            """)
        else:
            cursor.execute("""
                SELECT id, keyword, description, is_active, created_at, updated_at
                FROM keywords
                ORDER BY keyword ASC
            """)
        
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_keyword(self, keyword_id: int) -> Optional[Dict]:
        """Get a keyword by ID.
        
        Args:
            keyword_id: Keyword ID
            
        Returns:
            Keyword dictionary or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, keyword, description, is_active, created_at, updated_at
            FROM keywords
            WHERE id = ?
        """, (keyword_id,))
        
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def update_keyword(self, keyword_id: int, keyword: str = None, 
                      description: str = None, is_active: bool = None) -> tuple[bool, str]:
        """Update a keyword.
        
        Args:
            keyword_id: Keyword ID
            keyword: New keyword text (optional)
            description: New description (optional)
            is_active: Active status (optional)
            
        Returns:
            Tuple of (success, message)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            updates = []
            params = []
            
            if keyword is not None:
                updates.append("keyword = ?")
                params.append(keyword.strip())
            if description is not None:
                updates.append("description = ?")
                params.append(description.strip() if description and description.strip() else None)
            if is_active is not None:
                updates.append("is_active = ?")
                params.append(1 if is_active else 0)
            
            if not updates:
                conn.close()
                return False, "❌ No updates provided"
            
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(keyword_id)
            
            cursor.execute(f"""
                UPDATE keywords
                SET {', '.join(updates)}
                WHERE id = ?
            """, params)
            
            conn.commit()
            conn.close()
            return True, "✅ Keyword updated successfully"
        except sqlite3.IntegrityError:
            conn.close()
            return False, "❌ Keyword already exists"
        except Exception as e:
            conn.close()
            return False, f"❌ Error updating keyword: {str(e)}"
    
    def delete_keyword(self, keyword_id: int) -> tuple[bool, str]:
        """Delete a keyword.
        
        Args:
            keyword_id: Keyword ID
            
        Returns:
            Tuple of (success, message)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM keywords WHERE id = ?", (keyword_id,))
            conn.commit()
            conn.close()
            return True, "✅ Keyword deleted successfully"
        except Exception as e:
            conn.close()
            return False, f"❌ Error deleting keyword: {str(e)}"
    
    def get_active_keywords(self) -> List[str]:
        """Get list of active keywords.
        
        Returns:
            List of keyword strings
        """
        keywords = self.get_keywords(active_only=True)
        return [kw['keyword'] for kw in keywords]
