import os
import requests
import json

class OllamaClient:
    """
    Client for interacting with the Ollama API to use Llama 3.1 model
    """
    
    def __init__(self, base_url="http://localhost:11434", model="llama3.1:latest"):
        """
        Initialize the Ollama client
        
        Args:
            base_url (str): Ollama API base URL
            model (str): Model name to use
        """
        self.base_url = base_url
        self.model = model
        
    def generate(self, prompt, context=None, system_prompt=None, temperature=0.7, max_tokens=2048):
        """
        Generate a response from the Ollama model
        
        Args:
            prompt (str): The prompt to send to the model
            context (list, optional): Context from previous interactions
            system_prompt (str, optional): System prompt for the model
            temperature (float): Temperature for generation (0.0 to 1.0)
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            dict: Response from the model
        """
        url = f"{self.base_url}/api/generate"
        
        # Prepare the payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Add optional parameters if provided
        if context:
            payload["context"] = context
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            # Make the API request
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return {"error": str(e)}
    
    def chat(self, messages, temperature=0.7, max_tokens=2048):
        """
        Generate a response using the Ollama chat endpoint
        
        Args:
            messages (list): List of message objects with 'role' and 'content'
            temperature (float): Temperature for generation (0.0 to 1.0)
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            dict: Response from the model
        """
        url = f"{self.base_url}/api/chat"
        
        # Prepare the payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            # Make the API request
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama chat API: {e}")
            return {"error": str(e)}
    
    def answer_with_context(self, question, context, system_prompt=None):
        """
        Generate an answer to a question using provided context
        
        Args:
            question (str): The question to answer
            context (list): List of context strings to inform the answer
            system_prompt (str, optional): System prompt for the model
            
        Returns:
            str: Generated answer
        """
        # Format context into a single string
        context_text = "\n\n".join(context)
        
        # Create a prompt that includes both context and question
        prompt = f"""Context information:
{context_text}

Based on the above context, please answer the following question:
{question}

If the question cannot be answered based on the provided context, please indicate that.
"""
        
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that accurately answers questions based only on the provided context."
        
        # Generate a response
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt
        )
        
        # Extract and return the answer
        if "error" in response:
            return f"Error generating response: {response['error']}"
        
        return response.get("response", "No response generated")
    
    def list_models(self):
        """
        List available models in Ollama
        
        Returns:
            list: Available models
        """
        url = f"{self.base_url}/api/tags"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            return data.get("models", [])
        except requests.exceptions.RequestException as e:
            print(f"Error listing Ollama models: {e}")
            return []