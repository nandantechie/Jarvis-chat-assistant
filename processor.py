import io
import os
from datetime import datetime
from typing import List, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from src.config import Config
import PyPDF2

class PDFProcessor:
    """
    Processes PDF documents using LangChain components for document loading and text splitting.
    Uses PyPDF2 for PDF reading and RecursiveCharacterTextSplitter for text chunking.
    """
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extracts text from a PDF file using PyPDF2.
        
        Args:
            pdf_file: A file path string, file-like object, or uploaded file
            
        Returns:
            str: The extracted text from the PDF
        """
        try:
            text = ""
            
            # Handle file path string
            if isinstance(pdf_file, str):
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            
            # Handle file-like objects (uploaded files)
            elif hasattr(pdf_file, 'read'):
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            return text.strip()
            
        except Exception as e:
            print(f"❌ Error extracting text from PDF: {str(e)}")
            raise e

    def create_document_chunks(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Splits text into chunks and creates Document objects.
        
        Args:
            text: The text to split
            metadata: Optional metadata to add to documents
            
        Returns:
            List[Document]: A list of LangChain Document objects with text chunks
        """
        if not text or not text.strip():
            return []
            
        try:
            chunks = self.text_splitter.split_text(text)
            
            documents = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    doc_metadata = metadata.copy() if metadata else {}
                    doc_metadata.update({
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "chunk_size": len(chunk)
                    })
                    
                    documents.append(Document(
                        page_content=chunk.strip(),
                        metadata=doc_metadata
                    ))
            
            print(f"✅ Created {len(documents)} document chunks")
            return documents
            
        except Exception as e:
            print(f"❌ Error creating document chunks: {str(e)}")
            raise e

    def process_pdf(self, pdf_file) -> List[Document]:
        """
        Full PDF processing pipeline: extract text and split into chunks.
        This is the main method called by the Flask app.
        
        Args:
            pdf_file: A file path string, file-like object, or uploaded file
            
        Returns:
            List[Document]: A list of LangChain Document objects with text chunks
        """
        try:
            print(f"🔄 Processing PDF: {getattr(pdf_file, 'name', pdf_file)}")
            
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_file)
            
            if not text.strip():
                raise Exception("No text could be extracted from the PDF. The file might be image-based or corrupted.")
            
            print(f"✅ Extracted {len(text)} characters from PDF")
            
            # Get filename for metadata
            if hasattr(pdf_file, 'name'):
                filename = pdf_file.name
            elif isinstance(pdf_file, str):
                filename = os.path.basename(pdf_file)
            else:
                filename = 'unknown.pdf'

            # Create document chunks with metadata
            documents = self.create_document_chunks(
                text, 
                {
                    "source": filename, 
                    "file_type": "pdf",
                    "total_characters": len(text),
                    "processed_at": str(datetime.now())
                }
            )

            if not documents:
                raise Exception("No document chunks were created from the PDF")

            print(f"✅ Successfully processed {filename}: {len(documents)} chunks created")
            return documents
            
        except Exception as e:
            filename = getattr(pdf_file, 'name', str(pdf_file))
            print(f"❌ Error processing document {filename}: {str(e)}")
            raise e

    def process_document(self, pdf_file) -> List[Document]:
        """
        Alias for process_pdf for backward compatibility.
        """
        return self.process_pdf(pdf_file)
    
    def validate_pdf(self, pdf_file) -> bool:
        """
        Validate if the file is a proper PDF that can be processed.
        
        Args:
            pdf_file: PDF file to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if isinstance(pdf_file, str):
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    return len(pdf_reader.pages) > 0
            elif hasattr(pdf_file, 'read'):
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                return len(pdf_reader.pages) > 0
            return False
        except:
            return False
    
    def get_pdf_info(self, pdf_file) -> dict:
        """
        Get basic information about the PDF.
        
        Args:
            pdf_file: PDF file to analyze
            
        Returns:
            dict: PDF information
        """
        try:
            if isinstance(pdf_file, str):
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    return {
                        'pages': len(pdf_reader.pages),
                        'encrypted': pdf_reader.is_encrypted,
                        'filename': os.path.basename(pdf_file)
                    }
            elif hasattr(pdf_file, 'read'):
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                return {
                    'pages': len(pdf_reader.pages),
                    'encrypted': pdf_reader.is_encrypted,
                    'filename': getattr(pdf_file, 'name', 'unknown.pdf')
                }
        except Exception as e:
            return {
                'error': str(e),
                'pages': 0,
                'encrypted': False,
                'filename': 'error.pdf'
            }
