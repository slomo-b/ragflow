# backend/app/rag_system.py - Einfaches RAG System für ChromaDB Integration
"""
Simple RAG (Retrieval Augmented Generation) System
Delegates most functionality to ChromaDB Manager
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class RAGSystem:
    """Simple RAG system that works with ChromaDB"""
    
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("RAGFLOW_GOOGLE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("RAGFLOW_OPENAI_API_KEY")
        
        # Initialize Google AI if available
        if GOOGLE_AI_AVAILABLE and self.google_api_key:
            genai.configure(api_key=self.google_api_key)
            self.google_available = True
        else:
            self.google_available = False
        
        # Initialize OpenAI if available
        if OPENAI_AVAILABLE and self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.openai_available = True
        else:
            self.openai_available = False
    
    async def generate_response(
        self, 
        query: str, 
        context_documents: List[Dict[str, Any]] = None,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate AI response with RAG context"""
        
        if not context_documents:
            context_documents = []
        
        # Build context from documents
        context_text = ""
        if context_documents:
            context_text = "\n\n".join([
                f"Document: {doc.get('filename', 'Unknown')}\n{doc.get('content', '')}"
                for doc in context_documents[:5]  # Limit to top 5 documents
            ])
        
        # Prepare prompt
        if context_text:
            prompt = f"""Based on the following context documents, please answer the user's question:

CONTEXT:
{context_text}

QUESTION: {query}

Please provide a helpful and accurate answer based on the context provided. If the context doesn't contain enough information to answer the question, please say so."""
        else:
            prompt = f"""Please answer the following question:

QUESTION: {query}

Please provide a helpful and accurate answer."""
        
        # Generate response
        try:
            if self.google_available and model.startswith("gemini"):
                response_text = await self._generate_google_response(prompt, model, temperature)
            elif self.openai_available and model.startswith("gpt"):
                response_text = await self._generate_openai_response(prompt, model, temperature)
            else:
                # Fallback response
                response_text = f"I understand you're asking: '{query}'. However, no AI provider is currently configured. Please set up Google AI or OpenAI API keys."
            
            return {
                "response": response_text,
                "model": model,
                "temperature": temperature,
                "context_used": len(context_documents),
                "features_used": {
                    "rag_search": len(context_documents) > 0,
                    "ai_generation": True,
                    "context_integration": len(context_documents) > 0
                },
                "intelligence_metadata": {
                    "query_complexity": "medium" if len(query) > 50 else "simple",
                    "reasoning_depth": "contextual" if context_documents else "general",
                    "context_integration": "high" if len(context_documents) > 2 else "low"
                }
            }
            
        except Exception as e:
            return {
                "response": f"I apologize, but I encountered an error while generating a response: {str(e)}",
                "error": str(e),
                "model": model,
                "temperature": temperature
            }
    
    async def _generate_google_response(self, prompt: str, model: str, temperature: float) -> str:
        """Generate response using Google AI"""
        try:
            model_instance = genai.GenerativeModel(model)
            
            response = model_instance.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=1000,
                )
            )
            
            return response.text
            
        except Exception as e:
            raise Exception(f"Google AI error: {str(e)}")
    
    async def _generate_openai_response(self, prompt: str, model: str, temperature: float) -> str:
        """Generate response using OpenAI"""
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"OpenAI error: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get RAG system status"""
        return {
            "google_ai": {
                "available": self.google_available,
                "configured": bool(self.google_api_key)
            },
            "openai": {
                "available": self.openai_available,
                "configured": bool(self.openai_api_key)
            },
            "features": {
                "rag_search": True,
                "ai_generation": self.google_available or self.openai_available,
                "context_integration": True
            }
        }

# Convenience instance
rag_system = RAGSystem()

# Development helper
if __name__ == "__main__":
    import json
    import asyncio
    
    print("🤖 RAG System Status:")
    print(json.dumps(rag_system.get_status(), indent=2))
    
    # Test generation
    async def test_generation():
        try:
            result = await rag_system.generate_response(
                "What is the capital of France?",
                context_documents=[]
            )
            print("\n🧪 Test Generation:")
            print(f"Response: {result['response'][:100]}...")
            print(f"Model: {result['model']}")
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
    
    asyncio.run(test_generation())