from pymongo import MongoClient
from datetime import datetime
from typing import Optional, Dict, List
import bcrypt

class DatabaseManager:
    """
    Manages MongoDB connections and operations for storing conversation history and user authentication.
    """
    def __init__(self):
        """
        Initialize the database manager with MongoDB connection.
        
        Args:
            uri: MongoDB connection URI
            db_name: Name of the database
        """
        try:
            # Try to connect to MongoDB
            from src.config import Config
            self.client = MongoClient(Config.MONGODB_URI)
            self.db = self.client[Config.DATABASE_NAME]
            self.users_collection = self.db.users
            self.conversations_collection = self.db.conversations
            self.messages_collection = self.db.messages
            
            # Test connection
            self.client.admin.command('ping')
            print("✅ Connected to MongoDB successfully")
            self.mongodb_available = True
            
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            print("📝 Using in-memory storage instead")
            self.mongodb_available = False
            
            # Fallback to in-memory storage
            self.users = {
                'admin': {'password': self._hash_password('admin123'), 'name': 'Administrator'},
                'demo': {'password': self._hash_password('demo123'), 'name': 'Demo User'}
            }
            self.conversations = []
            self.messages = []

    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user credentials"""
        try:
            if self.mongodb_available:
                user = self.users_collection.find_one({"username": username})
                if user and self._verify_password(password, user['password']):
                    return {
                        'name': user.get('display_name', username),
                        'username': username,
                        'created_at': user.get('created_at')
                    }
            else:
                # In-memory fallback
                if username in self.users and self._verify_password(password, self.users[username]['password']):
                    return {
                        'name': self.users[username]['name'],
                        'username': username
                    }
            return None
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return None

    def register_user(self, username: str, password: str, display_name: str = None) -> bool:
        """Register a new user"""
        try:
            if self.user_exists(username):
                return False

            user_data = {
                'username': username,
                'password': self._hash_password(password),
                'display_name': display_name or username,
                'created_at': datetime.now()
            }

            if self.mongodb_available:
                self.users_collection.insert_one(user_data)
            else:
                # In-memory fallback
                self.users[username] = {
                    'password': user_data['password'],
                    'name': user_data['display_name']
                }
            
            print(f"✅ User {username} registered successfully")
            return True
        except Exception as e:
            print(f"❌ Registration error: {e}")
            return False

    def user_exists(self, username: str) -> bool:
        """Check if user exists"""
        try:
            if self.mongodb_available:
                return self.users_collection.find_one({"username": username}) is not None
            else:
                return username in self.users
        except Exception as e:
            print(f"❌ User check error: {e}")
            return False

    def save_conversation_message(self, username: str, conversation_id: str, role: str, content: str) -> bool:
        """Save a single message to conversation history"""
        try:
            message_data = {
                'username': username,
                'conversation_id': conversation_id,
                'role': role,  # 'user' or 'assistant'
                'content': content,
                'timestamp': datetime.now(),
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            if self.mongodb_available:
                self.messages_collection.insert_one(message_data)
            else:
                # In-memory fallback
                self.messages.append(message_data)

            print(f"💾 Saved message for {username} in conversation {conversation_id}")
            return True
        except Exception as e:
            print(f"❌ Error saving message: {e}")
            return False

    def get_conversation_history(self, username: str, conversation_id: str, limit: int = 50) -> List[Dict]:
        """Get conversation history for a user and conversation"""
        try:
            if self.mongodb_available:
                messages = list(self.messages_collection.find({
                    'username': username,
                    'conversation_id': conversation_id
                }).sort('timestamp', 1).limit(limit))
                
                # Convert ObjectId to string for JSON serialization
                for msg in messages:
                    msg['_id'] = str(msg['_id'])
                
                return messages
            else:
                # In-memory fallback
                messages = [
                    msg for msg in self.messages 
                    if msg['username'] == username and msg['conversation_id'] == conversation_id
                ]
                return sorted(messages, key=lambda x: x['timestamp'])[-limit:]
        except Exception as e:
            print(f"❌ Error getting conversation history: {e}")
            return []

    def get_user_conversations(self, username: str, limit: int = 20) -> List[Dict]:
        """Get list of conversations for a user"""
        try:
            if self.mongodb_available:
                # Get unique conversations with latest message
                pipeline = [
                    {'$match': {'username': username}},
                    {'$group': {
                        '_id': '$conversation_id',
                        'last_message': {'$last': '$content'},
                        'last_timestamp': {'$last': '$timestamp'},
                        'message_count': {'$sum': 1}
                    }},
                    {'$sort': {'last_timestamp': -1}},
                    {'$limit': limit}
                ]
                
                conversations = list(self.messages_collection.aggregate(pipeline))
                
                # Format for frontend
                formatted_conversations = []
                for conv in conversations:
                    formatted_conversations.append({
                        'conversation_id': conv['_id'],
                        'title': conv['last_message'][:50] + '...' if len(conv['last_message']) > 50 else conv['last_message'],
                        'last_message': conv['last_message'],
                        'last_timestamp': conv['last_timestamp'],
                        'message_count': conv['message_count'],
                        'created_at': conv['last_timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                return formatted_conversations
            else:
                # In-memory fallback
                user_messages = [msg for msg in self.messages if msg['username'] == username]
                conversations = {}
                
                for msg in user_messages:
                    conv_id = msg['conversation_id']
                    if conv_id not in conversations:
                        conversations[conv_id] = {
                            'conversation_id': conv_id,
                            'title': msg['content'][:50] + '...' if len(msg['content']) > 50 else msg['content'],
                            'last_message': msg['content'],
                            'last_timestamp': msg['timestamp'],
                            'message_count': 1,
                            'created_at': msg['created_at']
                        }
                    else:
                        conversations[conv_id]['last_message'] = msg['content']
                        conversations[conv_id]['last_timestamp'] = msg['timestamp']
                        conversations[conv_id]['message_count'] += 1
                
                return list(conversations.values())[:limit]
        except Exception as e:
            print(f"❌ Error getting conversations: {e}")
            return []

    def create_conversation(self, username: str, conversation_id: str = None) -> str:
        """Create a new conversation"""
        try:
            if not conversation_id:
                conversation_id = f"{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            conversation_data = {
                'conversation_id': conversation_id,
                'username': username,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }

            if self.mongodb_available:
                self.conversations_collection.insert_one(conversation_data)
            else:
                self.conversations.append(conversation_data)

            print(f"✅ Created conversation {conversation_id} for {username}")
            return conversation_id
        except Exception as e:
            print(f"❌ Error creating conversation: {e}")
            return None

    def delete_conversation(self, username: str, conversation_id: str) -> bool:
        """Delete a conversation and all its messages"""
        try:
            if self.mongodb_available:
                # Delete messages
                self.messages_collection.delete_many({
                    'username': username,
                    'conversation_id': conversation_id
                })
                
                # Delete conversation
                self.conversations_collection.delete_one({
                    'username': username,
                    'conversation_id': conversation_id
                })
            else:
                # In-memory fallback
                self.messages = [
                    msg for msg in self.messages 
                    if not (msg['username'] == username and msg['conversation_id'] == conversation_id)
                ]
                self.conversations = [
                    conv for conv in self.conversations 
                    if not (conv['username'] == username and conv['conversation_id'] == conversation_id)
                ]

            print(f"🗑️ Deleted conversation {conversation_id} for {username}")
            return True
        except Exception as e:
            print(f"❌ Error deleting conversation: {e}")
            return False

# Test connection function
def test_mongodb_connection():
    """Test MongoDB connection"""
    try:
        db_manager = DatabaseManager()
        if db_manager.mongodb_available:
            print("✅ MongoDB connection test successful")
            return True
        else:
            print("⚠️ Using in-memory storage")
            return False
    except Exception as e:
        print(f"❌ MongoDB connection test failed: {e}")
        return False

if __name__ == "__main__":
    test_mongodb_connection()
