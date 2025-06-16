import os
import logging
import requests
import json
from typing import Dict, Any, Optional, List

class MistralRunner:
    """Interface for running Mistral 7B model via API."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # API configuration
        # Check if config has models.mistral structure or flat structure
        if 'models' in self.config and 'mistral' in self.config['models']:
            mistral_config = self.config['models']['mistral']
            self.api_key = mistral_config.get('api_key', os.environ.get('MISTRAL_API_KEY', ''))
            self.api_base = mistral_config.get('api_base', 'https://api.mistral.ai/v1')
            self.model_name = mistral_config.get('model_name', 'mistral-small')
            self.temperature = mistral_config.get('temperature', 0.7)
            self.max_tokens = mistral_config.get('max_tokens', 10000)
        else:
            # Fallback to flat config structure
            self.api_key = self.config.get('mistral_api_key', os.environ.get('MISTRAL_API_KEY', ''))
            self.api_base = self.config.get('mistral_api_base', 'https://api.mistral.ai/v1')
            self.model_name = self.config.get('mistral_model', 'mistral-small')
            self.temperature = self.config.get('temperature', 0.7)
            self.max_tokens = self.config.get('max_tokens', 10000)
        
        # Check if API key is available
        self._check_api_access()
    
    def _check_api_access(self) -> None:
        """Check if API key is available and valid."""
        if not self.api_key:
            self.logger.warning("Mistral API key not found. Set it in config or MISTRAL_API_KEY environment variable.")
            return
        
        try:
            # Make a simple API call to check access
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(f"{self.api_base}/models", headers=headers)
            
            if response.status_code != 200:
                self.logger.warning(f"Mistral API access check failed: {response.text}")
            else:
                self.logger.info("Mistral API access confirmed.")
                
        except Exception as e:
            self.logger.warning(f"Error checking Mistral API access: {str(e)}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Mistral model via API."""
        if not self.api_key:
            return "Error: Mistral API key not configured. Please set MISTRAL_API_KEY environment variable or update config."
            
        try:
            # Override default parameters with any provided kwargs
            temperature = kwargs.get('temperature', self.temperature)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            
            # Prepare the API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Use the chat completions format instead of completions
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Make the API request
            self.logger.info(f"Calling Mistral API with prompt length: {len(prompt)}")
            response = requests.post(
                f"{self.api_base}/chat/completions", 
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                self.logger.error(f"Error from Mistral API: {response.text}")
                return f"Error: API returned status code {response.status_code}: {response.text}"
            
            # Parse the response - the structure is different in chat completions
            result = response.json()
            return result.get('choices', [{}])[0].get('message', {}).get('content', '')
            
        except Exception as e:
            self.logger.error(f"Error generating text with Mistral API: {str(e)}")
            return f"Error: {str(e)}"
    
    def summarize(self, text: str, length: str = 'medium') -> str:
        """Summarize text using Mistral model."""
        # Map length to token count
        length_map = {
            'short': 500,
            'medium': 900,
            'long': 1400
        }
        max_tokens = length_map.get(length, 900)
        
        # Create summarization prompt
        prompt = f"""Please summarize the following text in a clear and concise manner.
        Length: {length} (approximately {max_tokens} tokens)
        
        Text to summarize:
        {text}
        
        Summary:"""
        
        return self.generate(prompt, max_tokens=max_tokens)
    
    def answer_question(self, question: str, context: Optional[str] = None) -> str:
        """Answer a question using Mistral model, optionally with context."""
        if context:
            prompt = f"""Use the following information to answer the question.
            
            Information:
            {context}
            
            Question: {question}
            
            Answer:"""
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        
        return self.generate(prompt)