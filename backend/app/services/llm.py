from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from app.core.config import settings
from typing import List
from langchain_core.documents import Document

class LLMService:
    def __init__(self):
        self.clients = {}
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an intelligent assistant designed to answer questions based on the provided document context.
            
Context:
{context}

Question:
{question}

Instructions:
1. Answer the question specifically using the context provided.
2. If the answer is not contained in the context, say "I cannot find the answer in the provided documents."
3. Keep the answer concise and professional.
            
Answer:"""
        )

        # Initialize Gemini
        if settings.GEMINI_API_KEY:
            try:
                gemini_llm = ChatGoogleGenerativeAI(
                    model="gemini-flash-latest",
                    google_api_key=settings.GEMINI_API_KEY,
                    temperature=0,
                    convert_system_message_to_human=True
                )
                self.clients["gemini"] = self.prompt_template | gemini_llm
                print("✅ Gemini Loaded")
            except Exception as e:
                print(f"⚠️ Gemini Init Failed: {e}")

        # Initialize OpenAI
        if settings.OPENAI_API_KEY:
            try:
                openai_llm = ChatOpenAI(
                    model_name=settings.LLM_MODEL, 
                    temperature=0, 
                    openai_api_key=settings.OPENAI_API_KEY
                )
                self.clients["openai"] = self.prompt_template | openai_llm
                print("✅ OpenAI Loaded")
            except Exception as e:
                print(f"⚠️ OpenAI Init Failed: {e}")

    def generate_response(self, query: str, context_docs: List[Document], provider: str = "gemini") -> str:
        # Fallback logic
        if provider not in self.clients:
            # If requested provider is missing, try to find ANY available
            if self.clients:
                provider = list(self.clients.keys())[0]
            else:
                return "❌ Configuration Error: No AI engines are active. Check server logs/keys."

        chain = self.clients[provider]
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        try:
            response = chain.invoke({
                "context": context_text,
                "question": query
            })
            
            content = response.content
            if isinstance(content, list):
                return "".join([part if isinstance(part, str) else str(part) for part in content])
            return str(content)
        except Exception as e:
            return f"Error ({provider}): {str(e)}"
