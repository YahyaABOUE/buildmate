from crewai import Agent
from tools.classifier_tool import ClassifierTool

def create_nlp_agent():
    return Agent(
        role="NLP Classification Specialist",
        goal=(
            "Analyze the user's free-text request and classify it into "
            "use_case, device_type, and budget_tier using the NLP classifier tool. "
            "Always return the full JSON output from the tool without modification."
        ),
        backstory=(
            "You are an expert in natural language understanding specialized in "
            "interpreting PC building requests. You extract structured information "
            "from casual or technical user descriptions and return precise classifications."
        ),
        tools=[ClassifierTool()],
        verbose=True,
        allow_delegation=False
    )