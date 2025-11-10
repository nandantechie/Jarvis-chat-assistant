from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import os
import bcrypt
from werkzeug.utils import secure_filename
from datetime import datetime
# Add these imports
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# Add these new imports
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Simple Database Manager
class SimpleDatabaseManager:
    def __init__(self):
        print("🗄️ Initializing Simple Database...")
        
        # Create test users with plain passwords for testing
        self.users = {
            'admin': {
                'password': 'admin123',  # Plain text for testing
                'name': 'Administrator'
            },
            'demo': {
                'password': 'demo123',   # Plain text for testing
                'name': 'Demo User'
            }
        }
        
        self.messages = []
        self.conversations = []
        
        print("✅ Simple database initialized with test users:")
        print("   👤 admin / admin123")
        print("   👤 demo / demo123")
    
    def authenticate_user(self, username, password):
        print(f"🔐 Authenticating: '{username}' with password length {len(password)}")
        
        if username in self.users:
            stored_password = self.users[username]['password']
            print(f"🔍 Stored password: '{stored_password}', Input password: '{password}'")
            
            if password == stored_password:  # Simple string comparison for testing
                print(f"✅ Authentication successful for '{username}'")
                return {
                    'name': self.users[username]['name'],
                    'username': username
                }
            else:
                print(f"❌ Password mismatch for '{username}'")
        else:
            print(f"❌ User '{username}' not found")
        
        return None
    
    def register_user(self, username, password, display_name):
        if username in self.users:
            return False
        
        self.users[username] = {
            'password': password,  # Store plain text for testing
            'name': display_name or username
        }
        
        print(f"✅ User '{username}' registered")
        return True
    
    def user_exists(self, username):
        return username in self.users
    
    # Placeholder methods for conversation history
    def save_conversation_message(self, username, conversation_id, role, content):
        message_data = {
            'username': username,
            'conversation_id': conversation_id,
            'role': role,
            'content': content,
            'timestamp': datetime.now(),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.messages.append(message_data)
        return True
    
    def get_user_conversations(self, username):
        return []
    
    def create_conversation(self, username, conversation_id=None):
        if not conversation_id:
            conversation_id = f"{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return conversation_id
    
    def get_conversation_history(self, username, conversation_id):
        return []
    
    def delete_conversation(self, username, conversation_id):
        return True

class AIManager:
    """Handles AI operations for document processing and chat"""
    
    def __init__(self):
        print("🤖 Initializing AI Manager...")
        
        # Initialize Google Gemini
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("❌ No Google API key found!")
            self.gemini_available = False
        else:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                print("✅ Google Gemini initialized")
                self.gemini_available = True
            except Exception as e:
                print(f"❌ Gemini initialization failed: {e}")
                self.gemini_available = False
        
        # Initialize sentence transformer for embeddings
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence Transformer loaded")
            self.embeddings_available = True
        except Exception as e:
            print(f"❌ Embeddings initialization failed: {e}")
            self.embeddings_available = False
        
        # Storage for document embeddings
        self.document_chunks = []
        self.embeddings = None
        self.faiss_index = None
    
    def create_embeddings(self, document_chunks):
        """Create embeddings for document chunks"""
        if not self.embeddings_available:
            return False
        
        try:
            print(f"🔄 Creating embeddings for {len(document_chunks)} chunks...")
            
            # Extract text content
            texts = [chunk.page_content for chunk in document_chunks]
            
            # Create embeddings
            embeddings = self.embedding_model.encode(texts)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(embeddings.astype('float32'))
            
            # Store chunks and embeddings
            self.document_chunks = document_chunks
            self.embeddings = embeddings
            
            print(f"✅ Created embeddings with dimension {dimension}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating embeddings: {e}")
            return False
    
    def search_similar_documents(self, query, k=5):
        """Search for similar documents using embeddings"""
        if not self.embeddings_available or self.faiss_index is None:
            return []
        
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search similar documents
            distances, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
            
            # Return relevant chunks
            relevant_chunks = []
            for idx in indices[0]:
                if idx < len(self.document_chunks):
                    relevant_chunks.append(self.document_chunks[idx])
            
            print(f"🔍 Found {len(relevant_chunks)} relevant chunks for query")
            return relevant_chunks
            
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []
    
    def generate_response(self, query, relevant_docs):
        """Generate AI response using Gemini"""
        if not self.gemini_available:
            return "AI is currently unavailable. Please check your API configuration."
        
        try:
            # Prepare context from relevant documents
            context = ""
            if relevant_docs:
                context = "\n".join([doc.page_content for doc in relevant_docs[:3]])  # Use top 3 chunks
            
            # Create prompt
            prompt = f"""You are a helpful AI assistant that answers questions based on the provided context from uploaded PDF documents.

Context from uploaded documents:
{context}

User Question: {query}

Instructions:
- Answer the question based on the context provided
- If the answer is not in the context, say so politely
- Be concise but informative
- If no context is provided, mention that no documents were found

Answer:"""

            # Generate response
            response = self.model.generate_content(prompt)
            
            return response.text
            
        except Exception as e:
            print(f"❌ Error generating AI response: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"

# Initialize managers
try:
    from src.database_mongo import MongoDatabaseManager
    db_manager = MongoDatabaseManager()
    print(f"✅ Database initialized: {'MongoDB' if db_manager.mongodb_available else 'Fallback'}")
except Exception as e:
    print(f"❌ Database initialization failed: {e}")
    # Fallback to simple database
    db_manager = SimpleDatabaseManager()

ai_manager = AIManager()

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('chat'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        print(f"🔐 Login attempt: username='{username}', password='{password}'")
        
        if not username or not password:
            flash('Please enter both username and password', 'error')
            return render_template('login.html')
        
        user_info = db_manager.authenticate_user(username, password)
        
        if user_info:
            session['username'] = username
            session['user_info'] = user_info
            session['messages'] = []
            session['uploaded_files'] = []
            
            flash('Login successful!', 'success')
            return redirect(url_for('chat'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        display_name = request.form.get('display_name', '').strip() or username
        
        if not username or not password:
            flash('Username and password are required', 'error')
            return render_template('login.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('login.html')
        
        if db_manager.user_exists(username):
            flash('Username already exists', 'error')
            return render_template('login.html')
        
        if db_manager.register_user(username, password, display_name):
            flash('Account created successfully! Please login.', 'success')
        else:
            flash('Registration failed. Please try again.', 'error')
    
    return render_template('login.html')

@app.route('/chat')
def chat():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    return render_template('chat.html',
                         username=session['username'],
                         user_info=session.get('user_info', {}),
                         uploaded_files=session.get('uploaded_files', []),
                         conversations=[],
                         current_conversation_id=None,
                         messages=session.get('messages', []))

@app.route('/upload', methods=['POST'])
def upload():
    """Handle file upload with AI processing"""
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    print("📤 Upload request received")
    
    if 'files' not in request.files:
        return jsonify({'success': False, 'message': 'No files provided'})
    
    files = request.files.getlist('files')
    print(f"📁 Received {len(files)} files")
    
    if not files or all(f.filename == '' for f in files):
        return jsonify({'success': False, 'message': 'No valid files selected'})
    
    uploaded_files = session.get('uploaded_files', [])
    processed_files = []
    errors = []
    all_document_chunks = []
    
    for file in files:
        if file.filename == '':
            continue
            
        print(f"🔄 Processing file: {file.filename}")
        
        # Check if it's a PDF
        if not file.filename.lower().endswith('.pdf'):
            errors.append(f"{file.filename}: Only PDF files are allowed")
            continue
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # Save file
            file.save(filepath)
            print(f"💾 Saved file: {filepath}")
            
            # Check file size
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                errors.append(f"{filename}: File is empty")
                os.remove(filepath)
                continue
            
            # Process PDF
            text_content = extract_text_from_pdf(filepath)
            
            if text_content and len(text_content.strip()) > 0:
                # Create chunks
                chunks = create_text_chunks(text_content, filename)
                all_document_chunks.extend(chunks)
                
                processed_files.append({
                    'name': filename,
                    'size': f"{len(chunks)} chunks",
                    'text_length': len(text_content)
                })
                
                print(f"✅ Successfully processed {filename}: {len(chunks)} chunks")
            else:
                errors.append(f"{filename}: No text could be extracted")
                os.remove(filepath)
                
        except Exception as e:
            error_msg = f"Error processing {filename}: {str(e)}"
            print(f"❌ {error_msg}")
            errors.append(error_msg)
            
            if os.path.exists(filepath):
                os.remove(filepath)
    
    # Create embeddings for all processed documents
    if all_document_chunks:
        print(f"🔄 Creating AI embeddings for {len(all_document_chunks)} total chunks...")
        
        embedding_success = ai_manager.create_embeddings(all_document_chunks)
        
        if embedding_success:
            print("✅ AI embeddings created successfully")
        else:
            print("⚠️ AI embeddings creation failed, but files were processed")
    
    # Update session
    if processed_files:
        uploaded_files.extend(processed_files)
        session['uploaded_files'] = uploaded_files
        
        message = f'Successfully processed {len(processed_files)} files with AI capabilities'
        if errors:
            message += f". Errors: {'; '.join(errors[:3])}"
        
        return jsonify({
            'success': True, 
            'message': message,
            'processed_files': len(processed_files),
            'total_chunks': len(all_document_chunks),
            'ai_enabled': ai_manager.gemini_available and ai_manager.embeddings_available,
            'errors': errors
        })
    
    elif errors:
        return jsonify({
            'success': False, 
            'message': f'Failed to process files: {"; ".join(errors[:3])}'
        })
    else:
        return jsonify({'success': False, 'message': 'No valid PDF files found'})

@app.route('/chat_message', methods=['POST'])
def chat_message():
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'success': False, 'message': 'Empty message'})
    
    # Check if files are uploaded
    uploaded_files = session.get('uploaded_files', [])
    
    if not uploaded_files:
        response = "Please upload some PDF documents first before asking questions!"
    else:
        # Use AI to generate intelligent response
        print(f"💬 Processing question: {message}")
        
        # Search for relevant documents
        relevant_docs = ai_manager.search_similar_documents(message, k=5)
        
        # Generate AI response
        response = ai_manager.generate_response(message, relevant_docs)
        
        print(f"🤖 AI Response generated: {len(response)} characters")
    
    # Save to session
    messages = session.get('messages', [])
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    messages.append({
        'role': 'user',
        'content': message,
        'timestamp': timestamp
    })
    
    messages.append({
        'role': 'assistant',
        'content': response,
        'timestamp': timestamp
    })
    
    session['messages'] = messages
    
    return jsonify({
        'success': True,
        'response': response,
        'timestamp': timestamp
    })

@app.route('/remove_file', methods=['POST'])
def remove_file():
    """Remove a file from session"""
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    data = request.get_json()
    filename = data.get('filename')
    
    uploaded_files = session.get('uploaded_files', [])
    uploaded_files = [f for f in uploaded_files if f['name'] != filename]
    session['uploaded_files'] = uploaded_files
    
    # Remove physical file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"🗑️ Removed file: {filepath}")
    
    return jsonify({'success': True})

@app.route('/clear_files', methods=['POST'])
def clear_files():
    """Clear all uploaded files"""
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    # Clear session files
    uploaded_files = session.get('uploaded_files', [])
    for file_info in uploaded_files:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_info['name'])
        if os.path.exists(filepath):
            os.remove(filepath)
    
    session['uploaded_files'] = []
    print("🧹 Cleared all files")
    
    return jsonify({'success': True})

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))

@app.route('/admin/stats')
def admin_stats():
    """Show database statistics (admin only)"""
    if 'username' not in session or session['username'] != 'admin':
        return redirect(url_for('login'))
    
    stats = db_manager.get_stats()
    
    return f"""
    <h1>ScrapMate Database Statistics</h1>
    <ul>
        <li>Status: {stats.get('status', 'unknown')}</li>
        <li>Users: {stats.get('users', 0)}</li>
        <li>Conversations: {stats.get('conversations', 0)}</li>
        <li>Messages: {stats.get('messages', 0)}</li>
        <li>Database: {stats.get('database_name', 'N/A')}</li>
    </ul>
    <p><a href="/chat">Back to Chat</a></p>
    """

@app.route('/new_conversation', methods=['POST'])
def new_conversation():
    """Start a new conversation"""
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    # Clear current messages
    session['messages'] = []
    session['current_conversation_id'] = None
    
    return jsonify({'success': True})

@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    """Clear current conversation"""
    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    # Clear current conversation
    session['messages'] = []
    current_conv_id = session.get('current_conversation_id')
    
    if current_conv_id:
        username = session['username']
        db_manager.delete_conversation(username, current_conv_id)
    
    session['current_conversation_id'] = None
    
    return jsonify({'success': True})

# Helper Functions for PDF Processing
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        print(f"📄 Extracted {len(text)} characters from PDF")
        return text.strip()
        
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {str(e)}")
        raise e

def create_text_chunks(text, filename, chunk_size=1000, chunk_overlap=200):
    """Create text chunks for processing"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        
        # Create Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                documents.append(Document(
                    page_content=chunk.strip(),
                    metadata={
                        "source": filename,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                ))
        
        print(f"📊 Created {len(documents)} text chunks for {filename}")
        return documents
        
    except Exception as e:
        print(f"❌ Error creating text chunks: {str(e)}")
        raise e

if __name__ == '__main__':
    print("🚀 Starting ScrapMate AI Chatbot...")
    print(f"👤 Test users: admin/admin123, demo/demo123")
    print("🌐 Server: http://localhost:5000")
    
    app.run(debug=True, port=5000)
