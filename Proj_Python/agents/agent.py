from google.adk.agents import Agent, LlmAgent, SequentialAgent, ParallelAgent
from google.adk.runners import Runner
from google.adk.tools import google_search
from google.genai import types
from google.adk.tools.tool_context import ToolContext


def agent_greeting(tool_context: ToolContext) -> dict:
    if not tool_context.state.get("greeted"):
        message = "Hello, I am a study assistant that gives you step-by-step instructions on solving your homework problems! Attach an image or input what you would like help with."
        tool_context.state["greeted"] = True
        return {"response": message}
    else:
        return {"response": ""}

code_greeting_agent = LlmAgent(
    #model="gemini-2.5-flash",
    name="GreetingAgent",
    description= "Greets users and tells them what it is capable of",
    instruction="You are a helpful assistant that greets the user and notifies them of what you are capable of. Use the \'agent_greeting\' tool to let the user know about your features. If the tool provides an empty response do not greet the user.",
    tools=[agent_greeting],
)

code_research_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="ResearchAgent",
    description="Researches and develops a solution to what the user needs help with",
    instruction="You are a research agent. Determine the steps used to solve the user's problem and provide those step-by-step solutions to the user. Use the Google Search tool to develop the solution and its steps. If given a mathematical formula or expression use Markdown syntax formatting for the output.",
    tools=[google_search],
    output_key= "research_output",
)

code_image_analysis_agent = LlmAgent(
    model="gemini-2.5-flash",
     name="ImageAnalysisAgent",
    description="Analyzes any image that the user inputs",
    instruction="You are an image analysis agent. You only run when the user attaches an image. Determine what the image that the user input is (if one is provided) and how it is relevant to their question. If given a mathematical formula or expression use Markdown syntax formatting for the output.",
    output_key= "image_analysis_output",
)

code_parallel_solution_agent = ParallelAgent(
    name="ParallelSolutionAgent",
    description="Runs the ResearchAgent and ImageAnalysisAgent to gather information for a solution to the user's problem",
    sub_agents=[code_research_agent,code_image_analysis_agent],
)

code_merger_agent = LlmAgent(
    name="MergerAgent",
    model="gemini-2.5-flash",
    description="Merges the information gathered from the ResearchAgent and ImageAnalysisAgent",
    instruction="""You are an agent that helps combine and structure the information gathered from the research and image analysis agents into a neat format. If given a mathematical formula or expression use Markdown syntax formatting for the output.
    
    Input from other Agents:
    -Topic Research: {research_output}
    -Image Analysis: {image_analysis_output}

    Provide a topic overview of what the question is about and then create a step-by-step solution to the problem for the user.
    """, 
)

code_workflow_agent = SequentialAgent(
    name='SQ_WorkflowAgent',
    description= "Executes the subagents in sequential order",
    sub_agents=[code_greeting_agent, code_parallel_solution_agent, code_merger_agent],
)

root_agent = code_workflow_agent