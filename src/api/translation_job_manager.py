"""Background Translation Job Manager.

Manages translation jobs that run in the background using threads.
Tracks progress and allows monitoring via API endpoints.
"""

import threading
import time
import logging
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime

# Configure logging for better visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TranslationJobManager:
    """Manages background translation jobs."""
    
    def __init__(self, books_db_manager):
        self.books_db = books_db_manager
        self.active_jobs: Dict[int, threading.Thread] = {}
        self.job_stats: Dict[int, Dict] = {}
        self.lock = threading.Lock()
    
    def start_translation_job(
        self,
        job_id: int,
        book_id: int,
        total_pages: int,
        ollama_model: str,
        ollama_base_url: str,
        use_refinement: bool,
        temperature: float,
        parallel_workers: int = 10
    ):
        """Start a background translation job in a separate thread."""
        with self.lock:
            if job_id in self.active_jobs:
                logger.warning(f"Job {job_id} is already running")
                return
            
            # Initialize job stats
            self.job_stats[job_id] = {
                'book_id': book_id,
                'total_pages': total_pages,
                'completed_pages': 0,
                'failed_pages': 0,
                'page_times': [],
                'start_time': time.time(),
                'last_start_time': time.time(),  # Track when this session started
                'total_processing_time': 0.0,  # Cumulative processing time (excluding stopped time)
                'parallel_workers': parallel_workers
            }
            
            # Start background thread
            thread = threading.Thread(
                target=self._run_translation_job,
                args=(job_id, book_id, total_pages, ollama_model, ollama_base_url, use_refinement, temperature, parallel_workers),
                daemon=True
            )
            thread.start()
            self.active_jobs[job_id] = thread
            logger.info(f"Started translation job {job_id} for book {book_id} in background thread")
    
    def _run_translation_job(
        self,
        job_id: int,
        book_id: int,
        total_pages: int,
        ollama_model: str,
        ollama_base_url: str,
        use_refinement: bool,
        temperature: float,
        parallel_workers: int
    ):
        """Run the translation job in background."""
        thread_start_time = time.time()
        logger.info(f"🔄 [Job {job_id}] ========== THREAD STARTED ==========")
        logger.info(f"🔄 [Job {job_id}] Thread ID: {threading.current_thread().ident}")
        logger.info(f"🔄 [Job {job_id}] Parameters: book_id={book_id}, pages={total_pages}, workers={parallel_workers}")
        
        try:
            # Update job status to 'running' at the start and record start time
            logger.info(f"📝 [Job {job_id}] Updating job status to 'running' in database...")
            from datetime import datetime
            current_time = datetime.now()
            self.books_db.update_job_progress(job_id, status='running', last_start_time=current_time)
            
            # Update in-memory stats with start time
            with self.lock:
                if job_id in self.job_stats:
                    self.job_stats[job_id]['last_start_time'] = time.time()
            logger.info(f"✅ [Job {job_id}] Job status updated to 'running'")
            
            logger.info(f"📦 [Job {job_id}] Importing required modules...")
            from langchain_ollama import OllamaLLM
            from src.api.main import extract_pdf_page_text_accurate
            logger.info(f"✅ [Job {job_id}] Modules imported successfully")
            
            # Get book info
            logger.info(f"📚 [Job {job_id}] Fetching book info from database (book_id={book_id})...")
            book = self.books_db.get_book(book_id)
            if not book:
                raise Exception(f"Book {book_id} not found")
            logger.info(f"✅ [Job {job_id}] Book found: {book.get('title', 'Unknown')}")
            
            pdf_path = Path(book['file_path'])
            logger.info(f"📄 [Job {job_id}] PDF path: {pdf_path}")
            if not pdf_path.exists():
                raise Exception(f"PDF file not found: {pdf_path}")
            logger.info(f"✅ [Job {job_id}] PDF file exists")
            
            # Initialize Ollama LLM once
            logger.info(f"🤖 [Job {job_id}] Initializing Ollama LLM...")
            logger.info(f"🤖 [Job {job_id}] Model: {ollama_model}, URL: {ollama_base_url}, Temp: {temperature}")
            llm_init_start = time.time()
            llm = OllamaLLM(
                model=ollama_model,
                base_url=ollama_base_url,
                temperature=temperature,
                top_p=0.9,
                num_ctx=4096
            )
            llm_init_time = time.time() - llm_init_start
            logger.info(f"✅ [Job {job_id}] Ollama LLM initialized successfully in {llm_init_time:.2f}s")
            
            # Process pages in parallel using thread pool
            import concurrent.futures
            from queue import Queue
            
            # Get already translated pages to skip them
            translated_pages = set(self.books_db.get_translated_pages(book_id))
            logger.info(f"📊 [Job {job_id}] Already translated pages: {len(translated_pages)}/{total_pages}")
            
            page_queue = Queue()
            for page_num in range(total_pages):
                # Skip pages that are already translated
                if page_num not in translated_pages:
                    page_queue.put(page_num)
                else:
                    logger.debug(f"⏭️ [Job {job_id}] Skipping already translated page {page_num + 1}")
            
            def translate_single_page(page_num: int):
                """Translate a single page."""
                page_start_time = time.time()
                page_display = page_num + 1
                try:
                    logger.info(f"📄 [Job {job_id}] [Page {page_display}/{total_pages}] ========== STARTING ==========")
                    
                    # Extract text
                    logger.info(f"📄 [Job {job_id}] [Page {page_display}] Extracting text from PDF...")
                    extract_start = time.time()
                    english_text = extract_pdf_page_text_accurate(pdf_path, page_num)
                    extract_time = time.time() - extract_start
                    logger.info(f"✅ [Job {job_id}] [Page {page_display}] Text extracted in {extract_time:.2f}s (length: {len(english_text) if english_text else 0} chars)")
                    
                    # Handle blank pages
                    if not english_text or not english_text.strip():
                        logger.info(f"📄 [Job {job_id}] [Page {page_display}] Page is blank - saving empty translation")
                        save_start = time.time()
                        self.books_db.save_translation(book_id, page_num, "", "")
                        save_time = time.time() - save_start
                        page_time = time.time() - page_start_time
                        logger.info(f"✅ [Job {job_id}] [Page {page_display}] Blank page saved in {save_time:.2f}s (total: {page_time:.2f}s)")
                        
                        with self.lock:
                            self.job_stats[job_id]['completed_pages'] += 1
                            self.job_stats[job_id]['page_times'].append(page_time)
                            self._update_job_progress(job_id)
                            logger.info(f"📊 [Job {job_id}] [Page {page_display}] Stats updated: completed={self.job_stats[job_id]['completed_pages']}, failed={self.job_stats[job_id]['failed_pages']}")
                        
                        return {'success': True, 'page_num': page_num, 'blank': True}
                    
                    # Use shared translation function
                    from src.utils.translation_utils import translate_text_with_ollama
                    
                    logger.info(f"🔄 [Job {job_id}] [Page {page_display}] Starting translation...")
                    translation_start = time.time()
                    hindi_translation = translate_text_with_ollama(
                        llm=llm,
                        english_text=english_text,
                        use_refinement=use_refinement,
                        page_display=f"Job {job_id} [Page {page_display}/{total_pages}]"
                    )
                    translation_time = time.time() - translation_start
                    logger.info(f"✅ [Job {job_id}] [Page {page_display}] Translation completed in {translation_time:.2f}s (length: {len(hindi_translation)} chars)")
                    
                    # Save translation
                    logger.info(f"💾 [Job {job_id}] [Page {page_display}] Saving translation to database...")
                    save_start = time.time()
                    success = self.books_db.save_translation(
                        book_id=book_id,
                        page_number=page_num,
                        original_text=english_text,
                        translated_text=hindi_translation
                    )
                    save_time = time.time() - save_start
                    page_time = time.time() - page_start_time
                    
                    logger.info(f"💾 [Job {job_id}] [Page {page_display}] Save result: {'SUCCESS' if success else 'FAILED'} (save time: {save_time:.2f}s, total: {page_time:.2f}s)")
                    
                    with self.lock:
                        if success:
                            self.job_stats[job_id]['completed_pages'] += 1
                            logger.info(f"📊 [Job {job_id}] [Page {page_display}] Incremented completed_pages")
                        else:
                            self.job_stats[job_id]['failed_pages'] += 1
                            logger.warning(f"📊 [Job {job_id}] [Page {page_display}] Incremented failed_pages (save failed)")
                        self.job_stats[job_id]['page_times'].append(page_time)
                        logger.info(f"📊 [Job {job_id}] [Page {page_display}] Added page_time: {page_time:.2f}s")
                        self._update_job_progress(job_id)
                        logger.info(f"📊 [Job {job_id}] [Page {page_display}] Progress updated in database")
                        current_stats = self.job_stats[job_id]
                        logger.info(f"📊 [Job {job_id}] [Page {page_display}] Current stats: completed={current_stats['completed_pages']}, failed={current_stats['failed_pages']}, total_times={len(current_stats['page_times'])}")
                    
                    logger.info(f"✅ [Job {job_id}] [Page {page_display}] ========== COMPLETED SUCCESSFULLY ========== (Total time: {page_time:.2f}s)")
                    return {'success': success, 'page_num': page_num}
                    
                except Exception as e:
                    page_time = time.time() - page_start_time
                    logger.error(f"❌ [Job {job_id}] [Page {page_display}] ========== FAILED ==========")
                    logger.error(f"❌ [Job {job_id}] [Page {page_display}] Error: {str(e)}", exc_info=True)
                    
                    with self.lock:
                        self.job_stats[job_id]['failed_pages'] += 1
                        self.job_stats[job_id]['page_times'].append(page_time)
                        logger.info(f"📊 [Job {job_id}] [Page {page_display}] Updated stats after failure")
                        self._update_job_progress(job_id)
                    
                    return {'success': False, 'page_num': page_num, 'error': str(e)}
            
            # Process pages with thread pool
            queue_size = page_queue.qsize()
            logger.info(f"🚀 [Job {job_id}] Starting to process {queue_size} pages with {parallel_workers} workers")
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {}
                pages_processed = 0
                
                # Submit initial batch of pages
                initial_submitted = 0
                while len(futures) < parallel_workers and not page_queue.empty():
                    page_num = page_queue.get()
                    future = executor.submit(translate_single_page, page_num)
                    futures[future] = page_num
                    initial_submitted += 1
                    logger.info(f"📤 [Job {job_id}] Submitted page {page_num + 1} for translation (initial batch: {initial_submitted}/{parallel_workers})")
                logger.info(f"✅ [Job {job_id}] Initial batch submitted: {initial_submitted} pages")
                
                # Process completed futures and submit new ones
                loop_iteration = 0
                last_progress_log = time.time()
                
                while futures or not page_queue.empty():
                    loop_iteration += 1
                    current_time = time.time()
                    
                    # Log loop status periodically (every 30 seconds)
                    if current_time - last_progress_log > 30:
                        logger.info(f"🔄 [Job {job_id}] Loop iteration {loop_iteration}: active_futures={len(futures)}, queue={page_queue.qsize()}, processed={pages_processed}")
                        last_progress_log = current_time
                    
                    if not futures:
                        # No active futures but queue not empty - submit more
                        if not page_queue.empty():
                            page_num = page_queue.get()
                            future = executor.submit(translate_single_page, page_num)
                            futures[future] = page_num
                            logger.info(f"📤 [Job {job_id}] Submitted page {page_num + 1} for translation (no active futures)")
                        else:
                            logger.info(f"✅ [Job {job_id}] No futures and queue empty - breaking loop")
                            break
                    
                    # Use as_completed with timeout to check for completed futures
                    # This is more reliable than wait() for long-running operations
                    try:
                        # Check if any futures are done (non-blocking check)
                        done_futures = [f for f in futures.keys() if f.done()]
                        
                        if done_futures:
                            logger.info(f"✅ [Job {job_id}] Found {len(done_futures)} completed future(s)")
                            # Process completed futures
                            for future in done_futures:
                                page_num = futures.pop(future)
                                pages_processed += 1
                                logger.info(f"📥 [Job {job_id}] Processing result for page {page_num + 1} (processed: {pages_processed})")
                                try:
                                    result = future.result(timeout=5.0)  # Increased timeout for getting result
                                    if result.get('success'):
                                        logger.info(f"✅ [Job {job_id}] Page {page_num + 1} result: SUCCESS")
                                    else:
                                        logger.warning(f"⚠️ [Job {job_id}] Page {page_num + 1} result: FAILED - {result.get('error', 'Unknown error')}")
                                except concurrent.futures.TimeoutError:
                                    logger.error(f"❌ [Job {job_id}] Timeout getting result for page {page_num + 1}")
                                except Exception as e:
                                    logger.error(f"❌ [Job {job_id}] Error getting result for page {page_num + 1}: {e}", exc_info=True)
                        else:
                            # No futures completed yet, wait a bit before checking again
                            time.sleep(0.5)
                            continue
                    except Exception as e:
                        logger.error(f"❌ [Job {job_id}] Error in main loop: {e}", exc_info=True)
                        time.sleep(1.0)
                        continue
                    
                    # Submit new pages if we have capacity and queue has items
                    submitted_new = 0
                    while len(futures) < parallel_workers and not page_queue.empty():
                        page_num = page_queue.get()
                        future = executor.submit(translate_single_page, page_num)
                        futures[future] = page_num
                        submitted_new += 1
                        logger.info(f"📤 [Job {job_id}] Submitted page {page_num + 1} for translation (new: {submitted_new})")
                    
                    if submitted_new > 0:
                        logger.info(f"✅ [Job {job_id}] Submitted {submitted_new} new pages (active: {len(futures)}, queue: {page_queue.qsize()})")
                    
                    # Update progress periodically
                    if pages_processed % 5 == 0 and pages_processed > 0:
                        with self.lock:
                            stats = self.job_stats.get(job_id, {})
                            logger.info(f"📊 [Job {job_id}] ========== PROGRESS UPDATE ==========")
                            logger.info(f"📊 [Job {job_id}] Pages processed: {pages_processed}/{queue_size}")
                            logger.info(f"📊 [Job {job_id}] Active futures: {len(futures)}")
                            logger.info(f"📊 [Job {job_id}] Queue remaining: {page_queue.qsize()}")
                            logger.info(f"📊 [Job {job_id}] Stats - completed: {stats.get('completed_pages', 0)}, failed: {stats.get('failed_pages', 0)}")
                            logger.info(f"📊 [Job {job_id}] Page times recorded: {len(stats.get('page_times', []))}")
                
                logger.info(f"✅ [Job {job_id}] All pages submitted. Waiting for remaining {len(futures)} futures to complete...")
                
                # Wait for all remaining futures to complete
                for future in concurrent.futures.as_completed(futures.keys()):
                    page_num = futures[future]
                    pages_processed += 1
                    try:
                        result = future.result()
                        if result.get('success'):
                            logger.debug(f"✅ [Job {job_id}] Page {page_num + 1} result: success")
                        else:
                            logger.warning(f"⚠️ [Job {job_id}] Page {page_num + 1} result: failed - {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        logger.error(f"❌ [Job {job_id}] Error getting result for page {page_num + 1}: {e}")
            
            # Job completed - final progress update
            thread_total_time = time.time() - thread_start_time
            logger.info(f"🏁 [Job {job_id}] ========== JOB COMPLETION PROCESS ==========")
            logger.info(f"🏁 [Job {job_id}] Thread execution time: {thread_total_time:.2f}s")
            
            with self.lock:
                stats = self.job_stats.get(job_id)
                if stats:
                    # Update total processing time (add this session's time)
                    session_time = time.time() - stats.get('last_start_time', time.time())
                    stats['total_processing_time'] = stats.get('total_processing_time', 0.0) + session_time
                    
                    completed = stats.get('completed_pages', 0)
                    failed = stats.get('failed_pages', 0)
                    page_times_count = len(stats.get('page_times', []))
                    total_processing_time = stats.get('total_processing_time', 0.0)
                    logger.info(f"📊 [Job {job_id}] Final stats from memory:")
                    logger.info(f"📊 [Job {job_id}]   - Completed pages: {completed}")
                    logger.info(f"📊 [Job {job_id}]   - Failed pages: {failed}")
                    logger.info(f"📊 [Job {job_id}]   - Page times recorded: {page_times_count}")
                    logger.info(f"📊 [Job {job_id}]   - Total processing time: {total_processing_time:.2f}s")
                    
                    # Final progress update
                    logger.info(f"💾 [Job {job_id}] Performing final progress update in database...")
                    self._update_job_progress(job_id)
                    logger.info(f"✅ [Job {job_id}] Final progress updated")
                else:
                    logger.warning(f"⚠️ [Job {job_id}] Job stats not found in memory!")
                    completed = 0
                    failed = 0
            
            # Mark job as completed in database
            logger.info(f"💾 [Job {job_id}] Marking job as 'completed' in database...")
            self.books_db.complete_job(job_id, 'completed')
            logger.info(f"✅ [Job {job_id}] Job marked as 'completed' in database")
            
            with self.lock:
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]
                    logger.info(f"🗑️ [Job {job_id}] Removed from active_jobs")
                else:
                    logger.warning(f"⚠️ [Job {job_id}] Job not found in active_jobs")
                if job_id in self.job_stats:
                    del self.job_stats[job_id]
                    logger.info(f"🗑️ [Job {job_id}] Removed from job_stats")
                else:
                    logger.warning(f"⚠️ [Job {job_id}] Job not found in job_stats")
            
            logger.info(f"✅ [Job {job_id}] ========== JOB COMPLETED SUCCESSFULLY ==========")
            logger.info(f"✅ [Job {job_id}] Total execution time: {thread_total_time:.2f}s")
                    
        except Exception as e:
            thread_total_time = time.time() - thread_start_time
            logger.error(f"❌ [Job {job_id}] ========== JOB FAILED ==========")
            logger.error(f"❌ [Job {job_id}] Error: {str(e)}", exc_info=True)
            logger.error(f"❌ [Job {job_id}] Thread execution time before failure: {thread_total_time:.2f}s")
            
            # Update total processing time before marking as failed
            with self.lock:
                stats = self.job_stats.get(job_id)
                if stats and 'last_start_time' in stats:
                    session_time = time.time() - stats['last_start_time']
                    stats['total_processing_time'] = stats.get('total_processing_time', 0.0) + session_time
                    # Save updated stats to database
                    self._update_job_progress(job_id)
            
            try:
                logger.info(f"💾 [Job {job_id}] Marking job as 'failed' in database...")
                self.books_db.complete_job(job_id, 'failed', str(e))
                logger.info(f"✅ [Job {job_id}] Job marked as 'failed' in database")
            except Exception as db_error:
                logger.error(f"❌ [Job {job_id}] Failed to update database: {db_error}", exc_info=True)
            
            with self.lock:
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]
                    logger.info(f"🗑️ [Job {job_id}] Removed from active_jobs after failure")
                if job_id in self.job_stats:
                    del self.job_stats[job_id]
                    logger.info(f"🗑️ [Job {job_id}] Removed from job_stats after failure")
            
            logger.error(f"❌ [Job {job_id}] ========== JOB FAILURE HANDLED ==========")
    
    def _update_job_progress(self, job_id: int):
        """Update job progress in database."""
        stats = self.job_stats.get(job_id)
        if not stats:
            return
        
        # Get all page times (from current session and previous sessions)
        page_times = stats.get('page_times', [])
        previous_page_times = stats.get('previous_page_times', [])
        
        # Combine previous and current page times
        all_page_times = previous_page_times + page_times
        
        # Also check database for any additional page times (in case database has more recent data)
        job = self.books_db.get_translation_job(job_id)
        if job and job.get('page_times'):
            import json
            try:
                db_page_times = json.loads(job['page_times'])
                # Use database times if they're longer (more up-to-date), otherwise use combined
                if len(db_page_times) >= len(all_page_times):
                    all_page_times = db_page_times
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Calculate average time per page from ALL processed pages
        # Only count pages that actually took time to process (exclude very short times)
        if all_page_times:
            # Filter out very short times (less than 0.1s) which might be blank pages or errors
            meaningful_times = [t for t in all_page_times if t >= 0.1]
            if meaningful_times:
                # Calculate average from meaningful processing times
                avg_time = sum(meaningful_times) / len(meaningful_times)
            else:
                # If all times are very short, use the overall average
                avg_time = sum(all_page_times) / len(all_page_times) if all_page_times else 0
        else:
            # No page times yet
            avg_time = 0
        
        # Calculate estimated time remaining based on average from processed pages
        completed = stats.get('completed_pages', 0) + stats.get('failed_pages', 0)
        total_pages = stats.get('total_pages', 0)
        remaining = max(0, total_pages - completed)
        
        # Only calculate estimate if we have meaningful timing data and remaining pages
        if avg_time > 0 and remaining > 0:
            est_time_remaining = avg_time * remaining
        else:
            est_time_remaining = 0
        
        # Get total processing time from stats (already includes previous sessions)
        # When job is running, we'll calculate it dynamically in get_job_progress
        # Here we just store the base value (from previous sessions)
        total_processing_time = stats.get('total_processing_time', 0.0)
        
        # Update database with all stats including page_times
        self.books_db.update_job_progress(
            job_id=job_id,
            completed_pages=stats.get('completed_pages', 0),
            failed_pages=stats.get('failed_pages', 0),
            avg_time_per_page=avg_time,
            estimated_time_remaining=est_time_remaining,
            page_times=all_page_times,  # Store all page times (current + previous)
            total_processing_time=total_processing_time
        )
        
        logger.debug(f"📊 [Job {job_id}] Progress update: completed={stats.get('completed_pages', 0)}, failed={stats.get('failed_pages', 0)}, avg_time={avg_time:.2f}s, est_remaining={est_time_remaining:.2f}s, total_processing_time={total_processing_time:.2f}s")
    
    def get_job_progress(self, job_id: int) -> Optional[Dict]:
        """Get current progress for a job."""
        job = self.books_db.get_translation_job(job_id)
        if not job:
            return None
        
        stats = self.job_stats.get(job_id, {})
        
        # Use in-memory stats if available (more up-to-date), otherwise use DB values
        completed_pages = stats.get('completed_pages', job.get('completed_pages', 0))
        failed_pages = stats.get('failed_pages', job.get('failed_pages', 0))
        
        # Get all page times (from current session and database)
        current_page_times = stats.get('page_times', [])
        all_page_times = current_page_times
        
        # Load previous page times from database
        if job.get('page_times'):
            import json
            try:
                previous_times = json.loads(job['page_times'])
                # Combine with current session times
                all_page_times = previous_times + current_page_times
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Calculate average time from ALL processed pages
        if all_page_times:
            # Filter out very short times (less than 0.1s) which might be blank pages or errors
            meaningful_times = [t for t in all_page_times if t >= 0.1]
            if meaningful_times:
                avg_time = sum(meaningful_times) / len(meaningful_times)
            else:
                # If all times are very short, use the overall average
                avg_time = sum(all_page_times) / len(all_page_times) if all_page_times else 0
        else:
            # Use database value if available
            avg_time = job.get('avg_time_per_page') or 0
        
        # Calculate estimated time remaining based on average from processed pages
        remaining = max(0, job['total_pages'] - completed_pages - failed_pages)
        if avg_time > 0 and remaining > 0:
            est_time_remaining = avg_time * remaining
        else:
            est_time_remaining = job.get('estimated_time_remaining') or 0
        
        # Calculate total processing time (including current session if running)
        # Base time from database (previous sessions)
        base_processing_time = job.get('total_processing_time') or 0.0
        
        # Add current session time if job is running
        if job_id in self.active_jobs and 'last_start_time' in stats:
            session_elapsed = time.time() - stats['last_start_time']
            total_processing_time = base_processing_time + session_elapsed
        else:
            total_processing_time = base_processing_time
        
        return {
            'success': True,
            'job_id': job_id,
            'book_id': job['book_id'],
            'status': job['status'],
            'total_pages': job['total_pages'],
            'completed_pages': completed_pages,
            'failed_pages': failed_pages,
            'avg_time_per_page': avg_time,
            'estimated_time_remaining': est_time_remaining,
            'total_processing_time': total_processing_time,
            'started_at': job.get('started_at'),
            'completed_at': job.get('completed_at'),
            'error_message': job.get('error_message')
        }
    
    def is_job_running(self, job_id: int) -> bool:
        """Check if a job is actually running in memory."""
        with self.lock:
            return job_id in self.active_jobs
    
    def resume_job(self, job_id: int) -> bool:
        """Resume a stuck job by restarting it from where it left off."""
        logger.info(f"🔄 [Job {job_id}] ========== RESUMING JOB ==========")
        
        with self.lock:
            # Check if job is already running
            if job_id in self.active_jobs:
                logger.warning(f"⚠️ [Job {job_id}] Job is already running in active_jobs!")
                return False
            logger.info(f"✅ [Job {job_id}] Job not in active_jobs, safe to resume")
            
            # Get job from database
            logger.info(f"📥 [Job {job_id}] Fetching job from database...")
            job = self.books_db.get_translation_job(job_id)
            if not job:
                logger.error(f"❌ [Job {job_id}] Job not found in database")
                return False
            logger.info(f"✅ [Job {job_id}] Job found in database: status={job.get('status')}, book_id={job.get('book_id')}")
            
            # Only resume if status is 'running' or 'pending' (stuck)
            if job['status'] not in ['running', 'pending']:
                logger.warning(f"⚠️ [Job {job_id}] Job has status '{job['status']}', cannot resume")
                return False
            
            # Get book info
            logger.info(f"📚 [Job {job_id}] Fetching book info (book_id={job['book_id']})...")
            book = self.books_db.get_book(job['book_id'])
            if not book:
                logger.error(f"❌ [Job {job_id}] Book {job['book_id']} not found")
                return False
            logger.info(f"✅ [Job {job_id}] Book found: {book.get('title', 'Unknown')}")
            
            # Initialize job stats with current progress from database
            completed_pages = job.get('completed_pages', 0)
            failed_pages = job.get('failed_pages', 0)
            total_processing_time = job.get('total_processing_time', 0.0) or 0.0
            
            # Load previous page times from database to preserve stats
            previous_page_times = []
            if job.get('page_times'):
                import json
                try:
                    previous_page_times = json.loads(job['page_times'])
                    logger.info(f"📊 [Job {job_id}] Loaded {len(previous_page_times)} previous page times from database")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"⚠️ [Job {job_id}] Could not parse page_times from database: {e}")
            
            logger.info(f"📊 [Job {job_id}] Resuming with: completed={completed_pages}, failed={failed_pages}, total_processing_time={total_processing_time:.2f}s, previous_page_times={len(previous_page_times)}")
            
            self.job_stats[job_id] = {
                'book_id': job['book_id'],
                'total_pages': job['total_pages'],
                'completed_pages': completed_pages,
                'failed_pages': failed_pages,
                'page_times': [],  # Start fresh for this session, but we'll combine with previous when calculating
                'previous_page_times': previous_page_times,  # Store previous times for calculation
                'start_time': time.time(),
                'last_start_time': time.time(),  # Track when this resume session started
                'total_processing_time': total_processing_time,  # Preserve cumulative processing time
                'parallel_workers': job.get('parallel_workers', 10)
            }
            logger.info(f"✅ [Job {job_id}] Job stats initialized for resume (preserving previous stats)")
            
            # Update job status to 'running' and record start time
            logger.info(f"💾 [Job {job_id}] Updating job status to 'running' in database...")
            from datetime import datetime
            current_time = datetime.now()
            self.books_db.update_job_progress(job_id, status='running', last_start_time=current_time)
            logger.info(f"✅ [Job {job_id}] Job status updated to 'running'")
            
            # Restart the job
            logger.info(f"🧵 [Job {job_id}] Creating new thread to resume job...")
            thread = threading.Thread(
                target=self._run_translation_job,
                args=(
                    job_id,
                    job['book_id'],
                    job['total_pages'],
                    job['ollama_model'],
                    job['ollama_base_url'],
                    job['use_refinement'],
                    job['temperature'],
                    job.get('parallel_workers', 10)
                )
            )
            self.active_jobs[job_id] = thread
            thread.daemon = True
            logger.info(f"🚀 [Job {job_id}] Starting thread...")
            thread.start()
            logger.info(f"✅ [Job {job_id}] Thread started successfully")
            
            logger.info(f"✅ [Job {job_id}] ========== JOB RESUMED SUCCESSFULLY ==========")
            logger.info(f"✅ [Job {job_id}] Already completed: {completed_pages}, failed: {failed_pages}")
            return True
    
    def cancel_job(self, job_id: int):
        """Cancel a running job."""
        with self.lock:
            if job_id in self.active_jobs:
                # Update total processing time before cancelling
                stats = self.job_stats.get(job_id)
                if stats and 'last_start_time' in stats:
                    session_time = time.time() - stats['last_start_time']
                    stats['total_processing_time'] = stats.get('total_processing_time', 0.0) + session_time
                    # Save updated stats to database
                    self._update_job_progress(job_id)
                
                # Mark as cancelled in database
                self.books_db.complete_job(job_id, 'cancelled')
                # Note: We can't actually stop the thread, but we mark it as cancelled
                del self.active_jobs[job_id]
                if job_id in self.job_stats:
                    del self.job_stats[job_id]
                logger.info(f"Job {job_id} marked as cancelled")
                return True
            return False

