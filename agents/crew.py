"""
Multi-Agent Analysis System using Groq
Lightweight implementation without heavy CrewAI dependencies
"""

import os
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

from config.settings import GROQ_MODEL
from storage.vector_store import VectorStore


class Agent:
    """Base agent class that uses Groq for LLM calls"""
    
    def __init__(self, name: str, role: str, goal: str, backstory: str, groq_client: Groq):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.client = groq_client
        
    def execute(self, task: str, context: str = "") -> str:
        """Execute a task with optional context"""
        system_prompt = f"""You are a {self.role}.

Your Goal: {self.goal}

Background: {self.backstory}

Provide detailed, professional, and actionable analysis."""

        user_prompt = f"""Task: {task}

{"Context from previous analysis:" + chr(10) + context if context else ""}

Please complete this task thoroughly."""

        print(f"\n{'='*60}")
        print(f"🤖 Agent: {self.name}")
        print(f"📋 Task: {task[:100]}...")
        print(f"{'='*60}")
        
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        result = response.choices[0].message.content
        print(f"✅ {self.name} completed task")
        return result


class NihongoNextCrew:
    """
    Orchestrates the multi-agent analysis of NihongoNext
    Using direct Groq SDK (no heavy dependencies)
    """
    
    def __init__(self, vector_store: VectorStore = None):
        self.vector_store = vector_store or VectorStore()
        self.client = self._init_groq()
        
        # Create agents
        self.business_analyst = Agent(
            name="Business Analyst",
            role="Senior Business Analyst",
            goal="Analyze NihongoNext's current offerings and identify strategic opportunities for growth, including target audience gaps and monetization channels.",
            backstory="You are a seasoned business analyst with 15 years of experience in the EdTech industry, specializing in language learning platforms. You excel at identifying market gaps and creating data-driven recommendations.",
            groq_client=self.client
        )
        
        self.it_architect = Agent(
            name="IT Solutions Architect",
            role="Senior IT Solutions Architect",
            goal="Examine technical infrastructure needs and propose a comprehensive AI roadmap with innovative features.",
            backstory="You are a veteran IT Solutions Architect with 12 years of experience building scalable EdTech platforms. You have deep expertise in AI/ML, NLP, and speech recognition for language learning.",
            groq_client=self.client
        )
        
        self.proposal_writer = Agent(
            name="Proposal Writer",
            role="Senior Technical Proposal Writer",
            goal="Synthesize business and technical recommendations into a compelling, professional Technical Proposal for the CEO.",
            backstory="You are an expert technical writer with a background in business and technology. Your proposals are known for clarity, strategic vision, and actionable recommendations.",
            groq_client=self.client
        )
        
    def _init_groq(self) -> Groq:
        """Initialize Groq client"""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Check your .env file.")
        
        print(f"Connecting to Groq with model: {GROQ_MODEL}")
        return Groq(api_key=api_key)
    
    def _get_context(self, query: str) -> str:
        """Get relevant context from vector store"""
        return self.vector_store.get_relevant_context(query, n_results=5)
    
    def run(self, user_query: str) -> str:
        """Run the full analysis pipeline"""
        print("\n" + "="*60)
        print("    Starting NihongoNext Multi-Agent Analysis")
        print("="*60)
        
        # Get context from vector store
        context = self._get_context(user_query)
        print(f"\n📚 Retrieved {len(context)} characters of context from vector store")
        
        # Task 1: Business Analysis
        business_task = f"""
Analyze NihongoNext's current business offerings and market position.

User Query: {user_query}

Website Content Context:
{context}

Based on this information, provide a comprehensive analysis including:
1. Summary of current offerings
2. Target audience gap analysis (business Japanese, JLPT prep levels, etc.)
3. Monetization opportunities with rationale
4. At least 3 specific growth recommendations

Be specific and data-driven.
"""
        business_analysis = self.business_analyst.execute(business_task)
        
        # Task 2: Technical Architecture
        tech_task = f"""
Based on the business analysis, propose a technical AI roadmap for NihongoNext.

User Query: {user_query}

Business Analysis Findings:
{business_analysis}

Website Content Context:
{context}

Propose innovative AI-driven features including:
- AI Japanese conversation partner
- Automated grading and feedback systems
- Spaced Repetition System (SRS) for vocabulary/kanji
- Personalized learning path optimization
- Voice recognition for pronunciation practice
- JLPT progress tracking and predictions

For each feature, provide:
- Technical requirements and architecture overview
- Implementation complexity (Low/Medium/High)
- Expected impact on student retention
- Recommended technology stack
"""
        tech_analysis = self.it_architect.execute(tech_task)
        
        # Task 3: Proposal Writing
        proposal_task = f"""
Synthesize all findings into a formal Technical Proposal for the CEO of NihongoNext.

User Query: {user_query}

Business Analysis:
{business_analysis}

Technical Recommendations:
{tech_analysis}

Structure the proposal with these sections:
1. Executive Summary
2. Current State Analysis
3. Market Opportunity Assessment
4. Proposed AI Feature Recommendations (with priority ranking)
5. Technical Implementation Roadmap
6. Expected ROI and Success Metrics
7. Next Steps and Call to Action

Format as professional markdown document.
Use clear, executive-friendly language.
Lead with value propositions and business impact.
End with a compelling call to action.
"""
        proposal = self.proposal_writer.execute(proposal_task)
        
        return proposal


def main():
    """Test the crew"""
    crew = NihongoNextCrew()
    result = crew.run(
        "Analyze NihongoNext and propose 3 AI-driven features to increase student retention."
    )
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(result)


if __name__ == "__main__":
    main()
