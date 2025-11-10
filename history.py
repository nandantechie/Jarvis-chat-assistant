import sqlite3
from datetime import datetime
import pytz
from src.config import Config

class UserHistoryManager:
    def __init__(self, db_path="user_history.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()
        self.migrate_existing_data()

    def create_tables(self):
        # Users table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT
            )
        """)

        # Conversations table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                title TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Messages table (replaces old history table)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                role TEXT,  -- 'user' or 'assistant'
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        """)

        # Legacy history table for migration
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                question TEXT,
                answer TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def _execute_and_get_id(self, query, params):
        self.cursor.execute(query, params)
        self.conn.commit()
        return self.cursor.lastrowid

    def signup(self, username, password):
        try:
            self.cursor.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, password)
            )
            self.conn.commit()
            return True, "Signup successful."
        except sqlite3.IntegrityError as e:
            if "UNIQUE" in str(e).upper():
                return False, "Username already exists."
            else:
                raise e

    def login(self, username, password):
        self.cursor.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        )
        return self.cursor.fetchone() is not None

    def migrate_existing_data(self):
        """Migrate existing Q&A data to the new conversation system"""
        try:
            self.cursor.execute("SELECT COUNT(*) FROM conversations")
            if self.cursor.fetchone()[0] > 0:
                return  # Already migrated

            self.cursor.execute("SELECT username, question, answer, timestamp FROM history ORDER BY username, timestamp")
            existing_data = self.cursor.fetchall()

            if not existing_data:
                return  # No data to migrate

            current_user = None
            conversation_id = None

            for username, question, answer, timestamp in existing_data:
                if username != current_user:
                    # Use timezone-aware datetime
                    now = datetime.now(pytz.timezone(Config.TIMEZONE))
                    conversation_id = self._execute_and_get_id(
                        "INSERT INTO conversations (username, title, created_at) VALUES (?, ?, ?)",
                        (username, f"Conversation {now.strftime('%Y-%m-%d %H:%M')}", timestamp)
                    )
                    current_user = username

                self.cursor.execute(
                    "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                    (conversation_id, "user", question, timestamp)
                )
                self.cursor.execute(
                    "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                    (conversation_id, "assistant", answer, timestamp)
                )

            self.conn.commit()
        except Exception as e:
            print(f"Migration error: {e}")

    def create_conversation(self, username, title=None):
        """Create a new conversation for the user"""
        if not title:
            # Use timezone-aware datetime for title
            now = datetime.now(pytz.timezone(Config.TIMEZONE))
            title = f"New Conversation {now.strftime('%Y-%m-%d %H:%M')}"

        now_str = datetime.now(pytz.timezone(Config.TIMEZONE)).strftime('%Y-%m-%d %H:%M:%S')
        self.cursor.execute(
            "INSERT INTO conversations (username, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (username, title, now_str, now_str)
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def update_conversation_timestamp(self, conversation_id):
        """Update the updated_at timestamp to current time"""
        now_str = datetime.now(pytz.timezone(Config.TIMEZONE)).strftime('%Y-%m-%d %H:%M:%S')
        self.conn.execute(
            "UPDATE conversations SET updated_at=? WHERE id=?",
            (now_str, conversation_id)
        )
        self.conn.commit()

    def get_conversations(self, username):
        """Get all conversations for a user"""
        cur = self.conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE username=? ORDER BY updated_at DESC",
            (username,)
        )
        return cur.fetchall()

    def save_message(self, conversation_id, role, content):
        """Save a message to a conversation"""
        self.conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content)
        )
        # Update conversation's updated_at timestamp
        self.conn.execute(
            "UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (conversation_id,)
        )
        self.conn.commit()

    def get_conversation_messages(self, conversation_id):
        """Get all messages for a conversation"""
        cur = self.conn.execute(
            "SELECT role, content, timestamp FROM messages WHERE conversation_id=? ORDER BY timestamp ASC",
            (conversation_id,)
        )
        return cur.fetchall()

    def update_conversation_title(self, conversation_id, title):
        """Update conversation title"""
        self.conn.execute(
            "UPDATE conversations SET title=? WHERE id=?",
            (title, conversation_id)
        )
        self.conn.commit()

    def delete_conversation(self, conversation_id):
        """Delete a conversation and all its messages"""
        self.conn.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
        self.conn.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        self.conn.commit()

    # Legacy methods for backward compatibility
    def save_history(self, username, question, answer):
        """Legacy method - creates a new conversation if none exists"""
        # Get or create default conversation for user
        cur = self.conn.execute(
            "SELECT id FROM conversations WHERE username=? ORDER BY updated_at DESC LIMIT 1",
            (username,)
        )
        result = cur.fetchone()

        if not result:
            conversation_id = self.create_conversation(username, "Default Conversation")
        else:
            conversation_id = result[0]

        self.save_message(conversation_id, "user", question)
        self.save_message(conversation_id, "assistant", answer)

    def fetch_history(self, username, limit=20):
        """Legacy method - returns messages from most recent conversation"""
        cur = self.conn.execute(
            "SELECT id FROM conversations WHERE username=? ORDER BY updated_at DESC LIMIT 1",
            (username,)
        )
        result = cur.fetchone()

        if not result:
            return []

        conversation_id = result[0]
        messages = self.get_conversation_messages(conversation_id)

        # Convert to legacy format (question, answer, timestamp)
        legacy_format = []
        user_msg = None
        for role, content, timestamp in messages:
            if role == "user":
                user_msg = (content, timestamp)
            elif role == "assistant" and user_msg:
                legacy_format.append((user_msg[0], content, user_msg[1]))
                user_msg = None

        return legacy_format[-limit:]
