from crewai import Agent
from tools.hardware_tool import HardwareTool

def create_recommendation_agent():
    return Agent(
        role="PC Hardware Recommendation Specialist",
        goal=(
            "Based on the classified use case, device type, budget tier, and maximum budget, "
            "query the hardware database and generate a complete and well-explained PC build "
            "or laptop recommendation. Always include explanations for every component selected."
        ),
        backstory=(
            "You are a seasoned PC hardware expert with deep knowledge of components, "
            "compatibility, and price-to-performance ratios. You always recommend builds "
            "that are practical, balanced, and tailored to the user's specific needs. "
            "You never recommend overkill components for simple tasks or underpowered "
            "components for demanding workloads."
        ),
        tools=[HardwareTool()],
        verbose=True,
        allow_delegation=False
    )