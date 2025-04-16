import base64
import tempfile
import os

class PDFHandler:
    """
    Class to handle PDF operations including text extraction and display
    """
    
    def extract_text(self, pdf_file):
        """
        Extract text from a PDF file
        
        Args:
            pdf_file: File object or path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        text = ""
        temp_file = None
        
        try:
            # Check if pdf_file is a file path or a file object
            if isinstance(pdf_file, str):
                # It's a file path
                file_path = pdf_file
            else:
                # It's a file object (like from Streamlit uploader)
                # Save to a temporary file
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, "temp.pdf")
                
                # Save the uploaded file to a temporary location
                with open(temp_path, "wb") as f:
                    f.write(pdf_file.read())
                
                # Reset file position if possible
                if hasattr(pdf_file, "seek"):
                    pdf_file.seek(0)
                
                file_path = temp_path
                temp_file = temp_path
            
            # Use PyPDF2 module if available
            try:
                import PyPDF2
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    num_pages = len(reader.pages)
                    
                    for page_num in range(num_pages):
                        page = reader.pages[page_num]
                        text += page.extract_text() + "\n\n"
            except ImportError:
                print("PyPDF2 not available, trying alternative method...")
                # Fallback to a simplified text extraction
                with open(file_path, 'rb') as file:
                    # Just read first few KB as a fallback
                    content = file.read(10000)
                    text = f"PDF content preview (PyPDF2 not available for full extraction): {content}"
            
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
        finally:
            # Clean up temporary file if created
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
                os.rmdir(os.path.dirname(temp_file))
    
    def display_pdf(self, pdf_file):
        """
        Generate HTML to display a PDF file using an iframe
        
        Args:
            pdf_file: File object or path to the PDF file
            
        Returns:
            str: HTML string for displaying the PDF
        """
        try:
            # Check if pdf_file is a file path or a file object
            if isinstance(pdf_file, str):
                # It's a file path
                with open(pdf_file, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            else:
                # It's a file object (like from Streamlit uploader)
                # First, check if we can get the name from the file
                if hasattr(pdf_file, "name"):
                    # This is for Streamlit uploaded files
                    base64_pdf = base64.b64encode(pdf_file.getvalue()).decode('utf-8')
                else:
                    # Try reading the content directly
                    pdf_content = pdf_file.read()
                    
                    # Reset file position if possible
                    if hasattr(pdf_file, "seek"):
                        pdf_file.seek(0)
                    
                    base64_pdf = base64.b64encode(pdf_content).decode('utf-8')
            
            # Embed PDF in HTML
            pdf_display = f"""
            <iframe 
                src="data:application/pdf;base64,{base64_pdf}" 
                width="100%" 
                height="600px" 
                type="application/pdf"
                style="border: 1px solid #ddd; border-radius: 5px;">
            </iframe>
            """
            return pdf_display
        except Exception as e:
            print(f"Error displaying PDF: {e}")
            return f"""
            <div style="border: 1px solid #ddd; border-radius: 5px; padding: 20px; text-align: center;">
                <p>Error displaying PDF: {str(e)}</p>
            </div>
            """
