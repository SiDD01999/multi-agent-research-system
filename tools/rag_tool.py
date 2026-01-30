"""
RAG Tool for CrewAI Agents
Provides access to the ChromaDB vector store for context retrieval
"""

from typing import Type, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from storage.vector_store import VectorStore


class RAGSearchInput(BaseModel):
    """Input schema for RAG search tool"""
    query: str = Field(
        description="The search query to find relevant information about NihongoNext"
    )
    num_results: int = Field(
        default=5,
        description="Number of relevant context chunks to retrieve"
    )


class RAGSearchTool(BaseTool):
    """
    Tool for searching the NihongoNext content vector store
    """
    name: str = "search_nihongonext_content"
    description: str = """
    Search the NihongoNext website content database.
    Use this tool to find specific information about:
    - Course offerings and curriculum
    - Pricing and packages
    - Teaching methodology
    - Instructor information
    - Company information and mission
    - Student testimonials and reviews
    
    Returns relevant text chunks from the website content.
    """
    args_schema: Type[BaseModel] = RAGSearchInput
    
    _vector_store: Optional[VectorStore] = None
    
    def __init__(self, vector_store: Optional[VectorStore] = None, **kwargs):
        super().__init__(**kwargs)
        self._vector_store = vector_store
        
    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store
    
    def _run(self, query: str, num_results: int = 5) -> str:
        """Execute the RAG search"""
        try:
            context = self.vector_store.get_relevant_context(
                query=query,
                n_results=num_results
            )
            return context
        except Exception as e:
            return f"Error searching content: {str(e)}"


def create_rag_tool(vector_store: VectorStore) -> RAGSearchTool:
    """Factory function to create RAG tool with injected vector store"""
    tool = RAGSearchTool(vector_store=vector_store)
    return tool
