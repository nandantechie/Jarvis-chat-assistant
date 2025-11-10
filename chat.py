from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage
from src.config import Config
import time

class ChatManager:
    """
    Manages chat interactions using LangChain components.
    Uses LangChain's ConversationalRetrievalChain for handling conversational RAG.
    """
    def __init__(self, api_key: str, db_manager=None):  # Fixed: changed parameter order and defaults
        """
        Initialize the ChatManager with the specified API key.
        
        Args:
            api_key: The API key for the LLM service
            db_manager: Database manager for persistent storage (optional)
        """
        self.api_key = api_key
        self.db_manager = db_manager
        self.memory = None
        self.chain = None
        self.llm = None
        self.retriever = None
        self._initialize_components()
        
    def _initialize_components(self):
        """
        Initialize LangChain components for the chat system.
        Sets up the LLM, memory, and creates the conversation chain.
        """
        try:
            # Initialize LLM
            self.llm = ChatGoogleGenerativeAI(
                model=Config.MODEL_NAME,
                google_api_key=self.api_key,
                temperature=getattr(Config, 'LLM_TEMPERATURE', 0.7),
                max_output_tokens=getattr(Config, 'MAX_TOKENS', 2048),
                convert_system_message_to_human=True
            )
            
            # Initialize memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            print("✅ Chat manager initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing chat manager: {e}")
            raise e

    def _create_chain(self, retriever):
        """Create conversational retrieval chain"""
        try:
            # Custom prompt template
            system_template = """You are a helpful AI assistant that answers questions based on the provided context from PDF documents.

Use the following pieces of context to answer the user's question. If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""

            qa_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=system_template
            )

            # Create the conversational retrieval chain
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": qa_prompt},
                verbose=False
            )
            
            print("✅ Conversational chain created successfully")
            
        except Exception as e:
            print(f"❌ Error creating chain: {e}")
            raise e

    def generate_response(self, query: str, context_docs: List[Document] = None) -> str:
        """
        Generate response using the conversational chain or direct LLM call
        
        Args:
            query: The user's question
            context_docs: Relevant documents for context (optional)
            
        Returns:
            str: The AI's response
        """
        try:
            if self.chain and self.retriever:
                # Use the conversational retrieval chain
                result = self.chain.invoke({"question": query})
                return result.get("answer", "I'm sorry, I couldn't generate a response.")
            
            else:
                # Fallback to direct LLM call with context
                if context_docs:
                    context_text = "\n".join([doc.page_content for doc in context_docs])
                    
                    messages = [
                        SystemMessage(content=f"""You are a helpful AI assistant. Use the following context to answer the user's question:

Context:
{context_text}

If the answer is not in the context, say so politely."""),
                        HumanMessage(content=query)
                    ]
                else:
                    messages = [
                        SystemMessage(content="You are a helpful AI assistant. Answer the user's question to the best of your ability."),
                        HumanMessage(content=query)
                    ]
                
                response = self.llm.invoke(messages)
                return response.content
                
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
                
    def set_retriever(self, retriever):
        """Set the retriever and create the conversational chain"""
        try:
            self.retriever = retriever
            if retriever:
                self._create_chain(retriever)
            print("✅ Retriever set successfully")
        except Exception as e:
            print(f"❌ Error setting retriever: {e}")
        
    def reset_conversation(self):
        """Reset the conversation memory"""
        try:
            if self.memory:
                self.memory.clear()
            print("✅ Conversation reset successfully")
        except Exception as e:
            print(f"❌ Error resetting conversation: {e}")

    def load_messages(self, user: str) -> list:
        """Load conversation history for a user"""
        try:
            if self.db_manager:
                messages = self.db_manager.load_conversation(user)
                return messages
            return []
        except Exception as e:
            print(f"❌ Error loading messages: {e}")
            return []

    def save_messages(self, user: str, messages: list):
        """Save conversation history for a user"""
        try:
            if self.db_manager:
                self.db_manager.save_conversation(user, messages)
            print("✅ Messages saved successfully")
        except Exception as e:
            print(f"❌ Error saving messages: {e}")

    def get_conversation_history(self):
        """Get current conversation history"""
        try:
            if self.memory:
                return self.memory.chat_memory.messages
            return []
        except Exception as e:
            print(f"❌ Error getting conversation history: {e}")
            return []
