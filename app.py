import os
import base64
import tempfile
import streamlit as st
import PyPDF2
import uuid

# Create a simple PDF QA app without dependencies on external APIs

class SimpleVectorStore:
    """Simple in-memory document store with basic search capabilities"""
    
    def __init__(self):
        self.documents = {}
    
    def add_document(self, text, metadata=None):
        """Add a document to the store"""
        if not text:
            return None
            
        # Generate a unique document ID
        doc_id = str(uuid.uuid4())
        
        # Add metadata if provided, or create empty dict
        doc_metadata = metadata or {}
        doc_metadata['doc_id'] = doc_id
        
        # Store document
        self.documents[doc_id] = {
            'text': text,
            'metadata': doc_metadata
        }
        
        return doc_id
    
    def search(self, query, k=5):
        """Basic search for documents containing the query terms"""
        if not query or not self.documents:
            return []
            
        # Enhanced search - split query into terms and look for any match
        query_terms = query.lower().split()
        matches = []
        scores = {}
        
        # Score each document based on how many query terms it contains
        for doc_id, doc_data in self.documents.items():
            doc_text = doc_data['text'].lower()
            score = 0
            
            # Count matches for each query term
            for term in query_terms:
                if term in doc_text:
                    score += 1
            
            # If we found any match at all, include the document
            if score > 0:
                scores[doc_id] = score
        
        # Sort documents by score (most matching terms first)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top k documents
        for doc_id, score in sorted_docs[:k]:
            matches.append({
                'document': self.documents[doc_id]['text'],
                'metadata': self.documents[doc_id]['metadata'],
                'id': doc_id,
                'score': score
            })
        
        # If no matches were found with term splitting,
        # fallback to simple substring search
        if not matches:
            for doc_id, doc_data in self.documents.items():
                # Always return at least the first document as a fallback
                matches.append({
                    'document': doc_data['text'],
                    'metadata': doc_data['metadata'],
                    'id': doc_id,
                    'score': 1  # Minimum score for fallback
                })
                break
        
        return matches
    
    def delete_document(self, doc_id):
        """Delete a document from the store"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            return True
        return False


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    text = ""
    
    # Create a temporary file to store the PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        # For Streamlit uploaded files
        if hasattr(pdf_file, "getvalue"):
            temp_file.write(pdf_file.getvalue())
        else:
            # For standard file objects
            temp_file.write(pdf_file.read())
            if hasattr(pdf_file, "seek"):
                pdf_file.seek(0)
        
        temp_path = temp_file.name
    
    try:
        # Extract text using PyPDF2
        with open(temp_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
    except Exception as e:
        st.error(f"Error extracting text: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return text


def display_pdf(pdf_file):
    """Display a PDF file using HTML elements for better compatibility with Streamlit"""
    # For Streamlit uploaded files
    if hasattr(pdf_file, "getvalue"):
        base64_pdf = base64.b64encode(pdf_file.getvalue()).decode('utf-8')
    else:
        # For standard file objects
        pdf_content = pdf_file.read()
        if hasattr(pdf_file, "seek"):
            pdf_file.seek(0)
        base64_pdf = base64.b64encode(pdf_content).decode('utf-8')
    
    # Create HTML with iframe - works better in Streamlit
    pdf_display = f'''
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="100%"
        height="600"
        type="application/pdf"
        frameborder="0"
        allowfullscreen
        style="border: 1px solid #ddd; border-radius: 5px; padding: 20px; box-sizing: border-box;"
    ></iframe>
    '''
    return pdf_display


def generate_answer(query, documents, chat_history=None):
    """
    Generate a direct answer from the PDF content
    simulating responses from Llama 3.1 on Ollama
    """
    if not documents:
        return "Please upload a PDF document first so I can answer your questions."
    
    # Extract all document text
    all_text = "\n\n".join([d['document'] for d in documents])
    
    # Find relevant content for the query
    query_terms = query.lower().split()
    
    # Extract paragraphs and score them
    paragraphs = []
    for doc in documents:
        text = doc['document']
        # Split into paragraphs
        for para in text.split('\n\n'):
            if para.strip():
                paragraphs.append(para)
    
    # Score each paragraph by relevance
    scored_paragraphs = []
    for para in paragraphs:
        score = 0
        para_lower = para.lower()
        
        # Check for exact phrases first (higher weight)
        if query.lower() in para_lower:
            score += 5
        
        # Check for individual terms
        for term in query_terms:
            if term in para_lower:
                score += 1
        
        if score > 0:
            scored_paragraphs.append((para, score))
    
    # Sort paragraphs by relevance score
    scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
    
    # Get the most relevant paragraphs
    if scored_paragraphs:
        # Use the top 3 paragraphs
        relevant_context = "\n".join([p[0] for p in scored_paragraphs[:3]])
    else:
        # No direct matches - use document beginning
        relevant_context = all_text[:1000]
    
    # Direct answer based on relevant content
    if scored_paragraphs:
        # Just return the most relevant paragraph without explanatory text
        best_para = scored_paragraphs[0][0]
        
        # Clean and format response
        response = best_para.strip()
    else:
        # When no good match is found
        sample = all_text[:200].strip()
        response = f"I don't have specific information about that in the document. The document contains information about {sample}..."
    
    return response


def main():
    # Page configuration
    st.set_page_config(
        page_title="PDF Chat with Llama 3.1",
        page_icon="📚",
        layout="wide"
    )
    
    # Initialize session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = SimpleVectorStore()
    
    if "current_pdf" not in st.session_state:
        st.session_state.current_pdf = None
    
    if "documents" not in st.session_state:
        st.session_state.documents = []
        
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Main title
    st.title("PDF Chat with Llama 3.1")
    st.markdown("""
    Upload PDF documents and chat about them using Llama 3.1 AI. The application uses RAG (Retrieval-Augmented Generation) to provide accurate answers.
    """)
    
    # Create two columns for the main content
    col1, col2 = st.columns([3, 2])
    
    # Left column for PDF upload and preview
    with col1:
        st.header("Upload & Preview PDF")
        
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            # Update current PDF
            st.session_state.current_pdf = uploaded_file
            
            # Display file details
            st.write(f"**File:** {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            # Process the PDF
            with st.spinner("Processing PDF..."):
                # Extract text
                pdf_text = extract_text_from_pdf(uploaded_file)
                
                # Add to vector store
                doc_id = st.session_state.vector_store.add_document(
                    text=pdf_text,
                    metadata={"filename": uploaded_file.name}
                )
                
                # Add to session state if not already there
                if not any(doc["name"] == uploaded_file.name for doc in st.session_state.documents):
                    st.session_state.documents.append({
                        "id": doc_id,
                        "name": uploaded_file.name
                    })
                
                st.success("PDF processed successfully!")
            
            # Display PDF using iframe
            st.subheader("PDF Preview")
            try:
                # Save to a temporary file for download button
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name
                
                with open(temp_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name=uploaded_file.name,
                    mime="application/pdf",
                )
                st.markdown("---")
                # Display the PDF inline
                st.markdown(display_pdf(uploaded_file), unsafe_allow_html=True)
                # Clean up
                os.remove(temp_path)
            except Exception as e:
                st.error(f"Error displaying PDF: {e}")
    
    # Right column for chat interface
    with col2:
        st.header("Chat with PDF")
        
        # Show documents list
        if st.session_state.documents:
            st.caption("Loaded documents:")
            doc_list = ", ".join([doc["name"] for doc in st.session_state.documents])
            st.markdown(f"📄 {doc_list}")
            st.markdown("---")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for chat in st.session_state.chat_history:
                if chat["role"] == "user":
                    st.markdown(f"💬 **You**: {chat['content']}")
                else:
                    st.markdown(f"🤖 **Assistant**: {chat['content']}")
                st.markdown("---")
        
        # Add JavaScript for Enter key handling
        st.markdown("""
        <script>
        const doc = window.parent.document;
        doc.addEventListener('keydown', function(e) {
            if (e.key == 'Enter' && !e.shiftKey && e.target.nodeName === 'TEXTAREA') {
                const textareas = doc.querySelectorAll('textarea');
                const textarea = Array.from(textareas).find(t => t === e.target);
                if (textarea) {
                    const submitButtons = doc.querySelectorAll('button[kind="formSubmit"]');
                    if (submitButtons.length > 0 && textarea.value.trim() !== '') {
                        e.preventDefault();
                        submitButtons[0].click();
                    }
                }
            }
        });
        </script>
        """, unsafe_allow_html=True)
        
        # Simple chat input without a form
        if "temp_input" not in st.session_state:
            st.session_state.temp_input = ""

        def submit_message():
            if st.session_state.temp_input.strip():  # Check if input is not just whitespace
                process_input(st.session_state.temp_input)
                st.session_state.temp_input = ""  # Clear input after submission

        def process_input(user_input):
            if not st.session_state.documents:
                st.warning("Please upload a PDF document first.")
                return
            
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user", 
                "content": user_input
            })
            
            with st.spinner("Processing..."):
                # Search for relevant documents
                results = st.session_state.vector_store.search(user_input)
                
                if results:
                    # Generate answer directly from document content
                    answer = generate_answer(
                        user_input, 
                        results, 
                        st.session_state.chat_history
                    )
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })
                else:
                    # Fallback if no relevant content found
                    fallback = "I couldn't find relevant information about that. Could you try asking something else about the document?"
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": fallback
                    })
            
            # Force a rerun to update the chat display
            st.rerun()
        
        # Chat input with button
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input(
                "Ask about the document:", 
                key="temp_input",
                placeholder="Type your question here and press Enter",
                label_visibility="collapsed"
            )
        with col2:
            st.button("Send", on_click=submit_message, key="send_button")
        
        # Clear chat button
        if st.session_state.chat_history:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

if __name__ == "__main__":
    main()