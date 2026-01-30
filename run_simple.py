"""
Autonomous Multi-Agent Research System
Run this file to analyze any website
"""

import os
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all configuration from single config file
from config import (
    TARGET_WEBSITE,
    WEBSITE_CONTENT,
    USER_QUERY,
    GROQ_MODEL,
    OUTPUT_DIR
)


class Agent:
    """AI Agent that uses Groq for LLM calls"""
    
    def __init__(self, name: str, role: str, goal: str, backstory: str, client: Groq):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.client = client
        
    def execute(self, task: str, context: str = "") -> str:
        """Execute a task"""
        system_prompt = f"""You are a {self.role}.
Your Goal: {self.goal}
Background: {self.backstory}
Provide detailed, professional, and actionable analysis."""

        user_prompt = f"""Task: {task}

{"Context from previous analysis:" + chr(10) + context if context else ""}

Please complete this task thoroughly."""

        print(f"\n{'='*60}")
        print(f"🤖 Agent: {self.name}")
        print(f"📋 Processing...")
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


def run_analysis():
    """Run the full multi-agent analysis"""
    print("\n" + "="*60)
    print("    Autonomous Multi-Agent Research System")
    print(f"    Target: {TARGET_WEBSITE}")
    print("="*60)
    
    # Initialize Groq client
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")
    
    client = Groq(api_key=api_key)
    print(f"\n✅ Connected to Groq ({GROQ_MODEL})")
    print(f"📚 Loaded {len(WEBSITE_CONTENT)} characters of content")
    
    # Create agents
    business_analyst = Agent(
        name="Business Analyst",
        role="Senior Business Analyst",
        goal="Analyze business offerings and identify strategic growth opportunities.",
        backstory="15 years experience in market analysis and business strategy.",
        client=client
    )
    
    it_architect = Agent(
        name="IT Solutions Architect", 
        role="Senior IT Solutions Architect",
        goal="Propose technical solutions and AI-driven feature roadmap.",
        backstory="12 years building scalable platforms with AI/ML expertise.",
        client=client
    )
    
    proposal_writer = Agent(
        name="Proposal Writer",
        role="Senior Technical Proposal Writer",
        goal="Create compelling, executive-ready technical proposals.",
        backstory="Expert at translating technical concepts into business value.",
        client=client
    )
    
    # ========== PHASE 1: Business Analysis ==========
    print("\n\n🔍 PHASE 1: Business Analysis")
    business_task = f"""
Analyze the following business/website:

Target: {TARGET_WEBSITE}

Content:
{WEBSITE_CONTENT}

User Query: {USER_QUERY}

Provide:
1. Summary of current offerings
2. Target audience gap analysis 
3. Monetization opportunities
4. At least 3 specific growth recommendations
"""
    business_analysis = business_analyst.execute(business_task)
    
    # ========== PHASE 2: Technical Architecture ==========
    print("\n\n🏗️ PHASE 2: Technical Architecture")
    tech_task = f"""
Based on the business analysis, propose a technical roadmap.

Target: {TARGET_WEBSITE}
User Query: {USER_QUERY}

Business Analysis:
{business_analysis}

Propose AI/tech features including (as applicable):
- Automation opportunities
- AI-powered features
- User experience improvements
- Scalability solutions

For each feature provide:
- Technical requirements
- Implementation complexity (Low/Medium/High)
- Expected business impact
- Recommended technology stack
"""
    tech_analysis = it_architect.execute(tech_task)
    
    # ========== PHASE 3: Proposal Writing ==========
    print("\n\n📝 PHASE 3: Writing Proposal")
    proposal_task = f"""
Create a formal Technical Proposal for the CEO/stakeholders.

Target: {TARGET_WEBSITE}

Business Analysis:
{business_analysis}

Technical Recommendations:
{tech_analysis}

Structure:
1. Executive Summary
2. Current State Analysis
3. Market Opportunity Assessment
4. Feature Recommendations (priority ranked)
5. Technical Implementation Roadmap
6. Expected ROI and Success Metrics
7. Next Steps and Call to Action

Format as professional markdown. Use executive-friendly language.
"""
    proposal = proposal_writer.execute(proposal_task)
    
    # ========== Save Output ==========
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Clean filename from website
    safe_name = TARGET_WEBSITE.replace("www.", "").replace(".", "_").replace("/", "")
    output_filename = f"{safe_name}_Business_Proposal.md"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    final_content = f"""# Business Proposal: {TARGET_WEBSITE}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Powered by:** Autonomous Multi-Agent Research System

---

{proposal}

---

*Generated using AI-powered multi-agent analysis.*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
    
    print("\n\n" + "="*60)
    print("    ✅ Analysis Complete!")
    print("="*60)
    print(f"\n📄 Proposal saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    try:
        run_analysis()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
