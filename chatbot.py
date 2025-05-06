import os
import requests

class Chatbot:
    """
    Class for handling the question-answering functionality using Ollama with Llama 3.1
    """
    
    def __init__(self, vector_store):
        """
        Initialize the chatbot with a vector store
        
        Args:
            vector_store: Instance of VectorStore class
        """
        # Store vector store reference
        self.vector_store = vector_store
        
        # Ollama API endpoint - , we need to modify the connection
        # For real deployment, this would be "http://localhost:11434"
        # we need to make Ollama accessible
        self.api_base = "https://your-ollama-service.com"  # This is a placeholder that needs to be replaced
        self.model = "llama3.1"  # Llama 3.1 model
    
    def answer_question(self, question, max_context_chunks=5):
        """
        Answer a question based on the content in the vector store
        
        Args:
            question (str): User question
            max_context_chunks (int): Maximum number of context chunks to include
            
        Returns:
            str: AI-generated answer
        """
        # For local development, we would check Ollama availability
        # we'll simulate a response
        # In a real deployment, you would need to ensure Ollama is accessible
        
        # Search for relevant chunks in the vector store
        relevant_chunks = self.vector_store.search(question, k=max_context_chunks)
        
        # Check if we have any relevant chunks
        if not relevant_chunks:
            return "I don't have any documents to answer your question. Please upload some PDFs first."
        
        # Extract document content
        chunks = []
        for item in relevant_chunks:
            if isinstance(item, dict):
                # Handle new vector store format
                if 'document' in item:
                    chunks.append(item['document'])
                elif 'text' in item:
                    chunks.append(item['text'])
            else:
                # Handle any other format
                chunks.append(str(item))
        
        # Build context from chunks
        context = "\n\n".join(chunks)
        
        # Create messages for the API call
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that answers questions based on the provided document context.
                Focus on the information in the context. If the answer cannot be found in the context, 
                acknowledge this clearly rather than making up information. If appropriate, include the source
                of information in your response. Always be clear, concise, and factual."""
            },
            {
                "role": "user",
                "content": f"""Context information is below:
                ---------------------
                {context}
                ---------------------
                
                Given the context information and not prior knowledge, answer the following question:
                {question}"""
            }
        ]
        
        try:
            # In a real environment, we would connect to Ollama
            # we'll simulate a response
            
            # Extract the question from the last message
            user_question = question
            
            # Generate a simple response based on the context
            if context:
                # Get first 500 characters of context for the simulation
                context_preview = context[:500] + "..." if len(context) > 500 else context
                
                # Simple simulation of an AI response
                simulated_response = (
                    f"Based on the document provided, I can see information about {context_preview.split()[:5]}...\n\n"
                    f"To properly answer your question about '{user_question}', I would need access to the Ollama API with Llama 3.1 model.\n\n"
                    f"In a real deployment, this would connect to Ollama running on localhost:11434. "
                    f"This is a simulation since we're running in Replit without Ollama access."
                )
                
                return simulated_response
            else:
                return "I couldn't find relevant information in the documents to answer your question."
                
        except Exception as e:
            return f"An error occurred: {str(e)}"
