"""
Business Analyst Agent for NihongoNext Analysis
"""

from crewai import Agent
from typing import List


def create_business_analyst(llm, tools: List) -> Agent:
    """
    Create the Business Analyst agent
    
    Role: Market and business opportunity analysis
    Goal: Identify target audience gaps and monetization opportunities
    """
    
    return Agent(
        role="Senior Business Analyst",
        goal="""Analyze NihongoNext's current offerings and identify strategic 
        opportunities for growth, including target audience gaps and new 
        monetization channels in the Japanese language learning market.""",
        backstory="""You are a seasoned business analyst with 15 years of 
        experience in the EdTech industry, specializing in language learning 
        platforms. You have helped numerous startups scale their offerings 
        and have deep knowledge of the Japanese language learning market, 
        including JLPT preparation, business Japanese, and conversational 
        courses. You excel at identifying market gaps and creating data-driven 
        recommendations for product expansion.""",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=True
    )
