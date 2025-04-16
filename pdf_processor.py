import os
import tempfile

class PDFProcessor:
    """
    Class to handle PDF operations including text extraction and chunking
    """
    
    def extract_text(self, pdf_file):
        """
        Extract text from a PDF file
        
        Args:
            pdf_file: File object (can be from FastAPI UploadFile)
            
        Returns:
            str: Extracted text content
        """
        text = ""
        
        # Create a temporary file to store the PDF
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "temp.pdf")
        
        try:
            # Save the content to a temporary file
            with open(temp_path, "wb") as f:
                f.write(pdf_file.read())
            
            # Try to use PyPDF2 if available
            try:
                import PyPDF2
                
                # Extract text from the saved PDF
                with open(temp_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    num_pages = len(reader.pages)
                    
                    for page_num in range(num_pages):
                        page = reader.pages[page_num]
                        text += page.extract_text() + "\n\n"
            except ImportError:
                print("PyPDF2 not available, trying alternative method...")
                
                # If PyPDF2 is not available, store file info instead
                file_stats = os.stat(temp_path)
                file_size = file_stats.st_size
                text = f"PDF Document uploaded (size: {file_size} bytes)\n\n"
                text += "The PDF content would be displayed here if PyPDF2 was available.\n"
                text += "This is a demonstration of the application's structure."
                
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            os.rmdir(temp_dir)
    
    def chunk_text(self, text, chunk_size=1000, chunk_overlap=200):
        """
        Split text into chunks for processing
        
        Args:
            text (str): Text to chunk
            chunk_size (int): Maximum size of each chunk
            chunk_overlap (int): Overlap between chunks
            
        Returns:
            list: List of text chunks
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end of chunk
            end = min(start + chunk_size, text_length)
            
            # If not at the end of text, find the last space to avoid splitting words
            if end < text_length:
                # Find the last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space != -1:
                    end = last_space
            
            # Add the chunk to our list
            chunks.append(text[start:end])
            
            # Move the start position, accounting for overlap
            start = max(start + chunk_size - chunk_overlap, end)
        
        return chunks