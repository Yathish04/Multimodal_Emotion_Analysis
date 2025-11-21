import sqlite3
import datetime
import os

DB_NAME = "emotion_data.db"

def init_db():
    """Initialize the database with the necessary table."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS emotion_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            modality TEXT NOT NULL,
            input_summary TEXT,
            predicted_emotion TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_emotion(modality, input_summary, predicted_emotion, confidence):
    """Log an emotion detection event."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        c.execute('''
            INSERT INTO emotion_logs (timestamp, modality, input_summary, predicted_emotion, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, modality, input_summary, predicted_emotion, confidence))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database Error: {e}")

def get_logs(limit=50):
    """Retrieve recent logs."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM emotion_logs ORDER BY id DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
