"""
Compatibility shim.

The project previously used `OllamaProcessor` from this module.
We now use Groq as the LLM backend, but keep this import path working.
"""

from models.groq_processor import GroqProcessor as OllamaProcessor
