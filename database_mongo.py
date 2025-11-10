from pymongo import MongoClient
from datetime import datetime
from typing import Optional, Dict, List
import bcrypt
import os
from dotenv import load_dotenv

load_dotenv()

class MongoDatabaseManager:
    def __init__(self):
        self.mongodb_available = False
        self.fallback_manager = None
        
        try:
            # Get MongoDB configuration from environment
            mongodb_uri = os.getenv('MONGODB_URI')
            database_name = os.getenv('DATABASE_NAME', 'scrapmate_chatbot')
            
            if not mongodb_uri:
                raise Exception("MONGODB_URI not found in environment variables")
            
            print("🔗 Connecting to MongoDB...")
            
            # Connect to MongoDB
            self.client = MongoClient(mongodb_uri)
            self.db = self.client[database_name]
            
            # Collections
            self.users_collection = self.db.users
            self.conversations_collection = self.db.conversations
            self.messages_collection = self.db.messages
            
            # Test connection
            self.client.admin.command('ping')
            print("✅ Connected to MongoDB successfully")
            print(f"📊 Database: {database_name}")
            
            # Create indexes for better performance
            self._create_indexes()
            
            self.mongodb_available = True
            
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            print("📝 Falling back to simple in-memory database")
            
            # Initialize fallback simple database
            self._init_fallback_database()
    
    def _create_indexes(self):
        """Create database indexes for better performance"""
        try:
            # Index for users
            self.users_collection.create_index("username", unique=True)
            
            # Indexes for messages
            self.messages_collection.create_index([("username", 1), ("conversation_id", 1)])
            self.messages_collection.create_index([("timestamp", -1)])
            
            # Indexes for conversations
            self.conversations_collection.create_index([("username", 1), ("conversation_id", 1)], unique=True)
            
            print("✅ Database indexes created")
        except Exception as e:
            print(f"⚠️ Index creation warning: {e}")
    
    def _init_fallback_database(self):
        """Initialize fallback simple database"""
        from app import SimpleDatabaseManager
        self.fallback_manager = SimpleDatabaseManager()
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except:
            return False

    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user credentials"""
        if not self.mongodb_available:
            return self.fallback_manager.authenticate_user(username, password)
        
        try:
            print(f"🔐 MongoDB: Authenticating user '{username}'")
            
            user = self.users_collection.find_one({"username": username})
            if user and self._verify_password(password, user['password_hash']):
                print(f"✅ MongoDB: Authentication successful for '{username}'")
                return {
                    'name': user.get('display_name', username),
                    'username': username,
                    'created_at': user.get('created_at')
                }
            else:
                print(f"❌ MongoDB: Authentication failed for '{username}'")
                return None
        except Exception as e:
            print(f"❌ MongoDB authentication error: {e}")
            return None

    def register_user(self, username: str, password: str, display_name: str = None) -> bool:
        """Register a new user"""
        if not self.mongodb_available:
            return self.fallback_manager.register_user(username, password, display_name)
        
        try:
            # Check if user already exists
            if self.user_exists(username):
                print(f"❌ MongoDB: User '{username}' already exists")
                return False

            user_data = {
                'username': username,
                'password_hash': self._hash_password(password),
                'display_name': display_name or username,
                'created_at': datetime.now(),
                'last_login': None
            }

            result = self.users_collection.insert_one(user_data)
            
            if result.inserted_id:
                print(f"✅ MongoDB: User '{username}' registered successfully")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ MongoDB registration error: {e}")
            return False

    def user_exists(self, username: str) -> bool:
        """Check if user exists"""
        if not self.mongodb_available:
            return self.fallback_manager.user_exists(username)
        
        try:
            return self.users_collection.find_one({"username": username}) is not None
        except Exception as e:
            print(f"❌ MongoDB user check error: {e}")
            return False

    def save_conversation_message(self, username: str, conversation_id: str, role: str, content: str) -> bool:
        """Save a single message to conversation history"""
        if not self.mongodb_available:
            return self.fallback_manager.save_conversation_message(username, conversation_id, role, content)
        
        try:
            message_data = {
                'username': username,
                'conversation_id': conversation_id,
                'role': role,  # 'user' or 'assistant'
                'content': content,
                'timestamp': datetime.now(),
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            result = self.messages_collection.insert_one(message_data)
            
            if result.inserted_id:
                print(f"💾 MongoDB: Saved message for {username} in conversation {conversation_id}")
                
                # Update conversation timestamp
                self.conversations_collection.update_one(
                    {'username': username, 'conversation_id': conversation_id},
                    {
                        '$set': {'updated_at': datetime.now()},
                        '$setOnInsert': {
                            'username': username,
                            'conversation_id': conversation_id,
                            'created_at': datetime.now()
                        }
                    },
                    upsert=True
                )
                
                return True
            return False
            
        except Exception as e:
            print(f"❌ MongoDB message save error: {e}")
            return False

    def get_conversation_history(self, username: str, conversation_id: str, limit: int = 50) -> List[Dict]:
        """Get conversation history for a user and conversation"""
        if not self.mongodb_available:
            return self.fallback_manager.get_conversation_history(username, conversation_id, limit)
        
        try:
            messages = list(self.messages_collection.find({
                'username': username,
                'conversation_id': conversation_id
            }).sort('timestamp', 1).limit(limit))
            
            # Convert ObjectId to string for JSON serialization
            for msg in messages:
                msg['_id'] = str(msg['_id'])
            
            print(f"📚 MongoDB: Retrieved {len(messages)} messages for conversation {conversation_id}")
            return messages
            
        except Exception as e:
            print(f"❌ MongoDB conversation history error: {e}")
            return []

    def get_user_conversations(self, username: str, limit: int = 20) -> List[Dict]:
        """Get list of conversations for a user with latest message preview"""
        if not self.mongodb_available:
            return self.fallback_manager.get_user_conversations(username, limit)
        
        try:
            # Aggregation pipeline to get conversations with latest message
            pipeline = [
                {'$match': {'username': username}},
                {'$sort': {'timestamp': -1}},
                {'$group': {
                    '_id': '$conversation_id',
                    'last_message': {'$first': '$content'},
                    'last_role': {'$first': '$role'},
                    'last_timestamp': {'$first': '$timestamp'},
                    'message_count': {'$sum': 1},
                    'created_at': {'$last': '$timestamp'}
                }},
                {'$sort': {'last_timestamp': -1}},
                {'$limit': limit}
            ]
            
            conversations = list(self.messages_collection.aggregate(pipeline))
            
            # Format for frontend
            formatted_conversations = []
            for conv in conversations:
                # Create title from first user message or truncate last message
                title = conv['last_message'][:50] + '...' if len(conv['last_message']) > 50 else conv['last_message']
                
                formatted_conversations.append({
                    'conversation_id': conv['_id'],
                    'title': title,
                    'last_message': conv['last_message'],
                    'last_timestamp': conv['last_timestamp'],
                    'message_count': conv['message_count'],
                    'created_at': conv['last_timestamp'].strftime('%Y-%m-%d %H:%M:%S') if conv['last_timestamp'] else ''
                })
            
            print(f"📋 MongoDB: Retrieved {len(formatted_conversations)} conversations for {username}")
            return formatted_conversations
            
        except Exception as e:
            print(f"❌ MongoDB conversations list error: {e}")
            return []

    def create_conversation(self, username: str, conversation_id: str = None) -> str:
        """Create a new conversation"""
        if not self.mongodb_available:
            return self.fallback_manager.create_conversation(username, conversation_id)
        
        try:
            if not conversation_id:
                conversation_id = f"{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            conversation_data = {
                'conversation_id': conversation_id,
                'username': username,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }

            self.conversations_collection.insert_one(conversation_data)
            print(f"✅ MongoDB: Created conversation {conversation_id} for {username}")
            return conversation_id
            
        except Exception as e:
            print(f"❌ MongoDB conversation creation error: {e}")
            return conversation_id  # Return the ID even if save failed

    def delete_conversation(self, username: str, conversation_id: str) -> bool:
        """Delete a conversation and all its messages"""
        if not self.mongodb_available:
            return self.fallback_manager.delete_conversation(username, conversation_id)
        
        try:
            # Delete messages
            messages_result = self.messages_collection.delete_many({
                'username': username,
                'conversation_id': conversation_id
            })
            
            # Delete conversation
            conv_result = self.conversations_collection.delete_one({
                'username': username,
                'conversation_id': conversation_id
            })
            
            print(f"🗑️ MongoDB: Deleted conversation {conversation_id} for {username}")
            print(f"   📧 Deleted {messages_result.deleted_count} messages")
            return True
            
        except Exception as e:
            print(f"❌ MongoDB conversation deletion error: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get database statistics"""
        if not self.mongodb_available:
            return {'status': 'fallback', 'users': len(self.fallback_manager.users)}
        
        try:
            stats = {
                'status': 'mongodb',
                'users': self.users_collection.count_documents({}),
                'conversations': self.conversations_collection.count_documents({}),
                'messages': self.messages_collection.count_documents({}),
                'database_name': self.db.name
            }
            return stats
        except Exception as e:
            print(f"❌ MongoDB stats error: {e}")
            return {'status': 'error', 'error': str(e)}

# Test connection function
def test_mongodb_connection():
    """Test MongoDB connection"""
    try:
        db_manager = MongoDatabaseManager()
        stats = db_manager.get_stats()
        print("📊 Database Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        return db_manager.mongodb_available
    except Exception as e:
        print(f"❌ MongoDB test failed: {e}")
        return False

if __name__ == "__main__":
    test_mongodb_connection()