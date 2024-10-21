# memory.py

import sqlite3
from datetime import datetime, timedelta
import threading
import time

DB_PATH = "memory.db"

# Memory retention durations
PERSISTENT_MEMORY_DURATION = None  # Never expire
SHORT_TERM_MEMORY_DURATION = timedelta(hours=1)  # Retain for 1 hour
MEMORY_CLEANUP_INTERVAL = 600  # Cleanup every 10 minutes

# Lock for thread-safe operations
memory_lock = threading.Lock()

def initialize_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS persistent_memory (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS short_term_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            command TEXT,
            response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def set_persistent(key, value):
    with memory_lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()
        c.execute('''
            INSERT INTO persistent_memory (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
        ''', (key, value))
        conn.commit()
        conn.close()

def get_persistent(key):
    with memory_lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()
        c.execute('SELECT value FROM persistent_memory WHERE key=?', (key,))
        result = c.fetchone()
        conn.close()
        return result[0] if result else None

def add_short_term(category, command, response):
    with memory_lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()
        c.execute('''
            INSERT INTO short_term_memory (category, command, response) VALUES (?, ?, ?)
        ''', (category, command, response))
        conn.commit()
        conn.close()

def get_short_term():
    with memory_lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()
        cutoff = datetime.now() - SHORT_TERM_MEMORY_DURATION
        c.execute('''
            SELECT category, command, response, timestamp 
            FROM short_term_memory 
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        ''', (cutoff,))
        results = c.fetchall()
        conn.close()
        return results

def cleanup_short_term():
    with memory_lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()
        cutoff = datetime.now() - SHORT_TERM_MEMORY_DURATION
        c.execute('DELETE FROM short_term_memory WHERE timestamp < ?', (cutoff,))
        conn.commit()
        conn.close()

def memory_cleanup_daemon():
    while True:
        cleanup_short_term()
        time.sleep(MEMORY_CLEANUP_INTERVAL)

def start_memory_cleanup():
    cleanup_thread = threading.Thread(target=memory_cleanup_daemon, daemon=True)
    cleanup_thread.start()
