import datetime
import os
from agents import Agent, AgentOutputSchema, FunctionToolResult, ItemHelpers, ModelSettings, RunHooks, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, ToolsToFinalOutputResult , handoff,Handoff,function_tool , AgentOutputSchemaBase, set_trace_processors , RunContextWrapper
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX , prompt_with_handoff_instructions
from agents.run import RunConfig
from dotenv import load_dotenv
import asyncio
from agents.tracing import set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
from dataclasses import dataclass
from langsmith.wrappers import OpenAIAgentsTracingProcessor
from typing import Any
from agents.extensions import handoff_filters
from pydantic import BaseModel
# set_tracing_disabled(True)
# set_trace_processors([OpenAIAgentsTracingProcessor()])
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")
set_trace_processors([OpenAIAgentsTracingProcessor()])
#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
    workflow_name='Quiz Prep Workflow 2',
)

# basic_agent = Agent(
#     name = "Basic Agent",
#     instructions="You are a basic agent that can answer questions.",
#     model = model
# )

#  EDGE CASES
#  CASE 1
#  Missing Instructions and name
# basic_agent = Agent(
#     name = "",
#     instructions="",
#     model = model
# )
# When instructions or name of the agent is missing , the agent will run as usual and the llm will provide a generic answer

#  CASE 2
# Providing with the wrong model name 
# basic_agent = Agent(
#     name = "Basic Agent",
#     instructions="You are a basic agent that can answer questions.",
#     model = model
# )
#  Not only Model name is not affecting anything , But also if we dont provide the model name , it will also run
#  This is becuase we are passing the config in the run config ,
# 
# 
#  If we remove the run Config from the Runner.run then we
#  have to provide the correct model  in the Agent class

# CASE 3
#  Handoffs
#  Creating two more agents to test the handoff
# math_agent = Agent(
#     name = "Math Agent",
#     instructions = "You are a math agent that can answer math related questions.",
#     model = model ,
#     handoff_description="Answer math related questions" # Description of the Agent when used in handoffs
    
# )

# computer_science_agent = Agent(
#     name = "Computer Science Agent",
#     instructions = "You are a computer science agent that can answer computer science related questions.",
#     model = model,
#     handoff_description="Answer computer science related questions" # Description of the Agent when used in handoffs
# )

# basic_agent = Agent(
#     name = "Basic Agent",
#     instructions="You are a basic agent that can answer basic questions , if the query is related to Math ,  Handoff to the Math agent and if the query is related to Computer Science , Handoff to the Computer Science agent",
#     model = model,
#     handoffs=[computer_science_agent , math_agent]
# )
# Missing a handoff agent but telling in the instructions , 
#  We pass the Math query and did not include in the handoffs of the agent but we have included in the instructions , and on the other hand we have passed the computer science agent , it handoffs to the Computer Science agent when any type of math or computer science related question is asked , Same for the other case , IF ONE AGENT IS NOT PROVIDED BUT INCLUDED IN THE INSTRUCTIONS THEN THE LLM WILL HANDOFF TO THE PROVIDED AGENT REGARDLESS OF IT IS BEING THE RIGHT AGENT.

# CASE 4    OUTPUT TYPE
# class AgentRespose(BaseModel):
#     response : dict[int , str]

# basic_agent = Agent(
#     name = "Basic Agent",
#     instructions="You are a basic agent that can answer questions.",
#     model = model,
#     output_type=AgentRespose
# )
#  When we pass a dict or a custom (complex) type of data type , then the agent raises an exception of 
# (agents.exceptions.UserError: Strict JSON schema is enabled, but the output type is not valid. Either make the output type strict, or pass output_schema_strict=False to your Agent()) , in our case dict[int , str]
#  TO RESOLVE THIS WE USE output_schema_strict=False

# class AgentRespose(BaseModel):
#     query : str
#     response : str
    
# basic_agent= Agent(
#     name = "Basic Agent",
#     instructions="You are a basic agent that can answer questions.",
#     model = model,
#     output_type=AgentRespose
# )

# CASE 5 , TOOL USE BEHAVIOUR
# @function_tool
# def get_current_time():
#     print('Getting current time')
#     return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# # Custom Behaviour of TOOLS
# def my_behavior(ctx: RunContextWrapper[None], tools: list[FunctionToolResult]) -> ToolsToFinalOutputResult:
#     if len(tools) >= 2:
#         return ToolsToFinalOutputResult(is_final_output= True ,final_output=tools[1].output)
#     return ToolsToFinalOutputResult(is_final_output= False ,continue_to_llm=True)

# @function_tool
# def get_current_date():
#     print('Getting current date')
#     return datetime.date.today().strftime("%Y-%m-%d")
# basic_agent= Agent(
#     name = "Basic Agent",
#     instructions="You are a basic agent that can answer questions. you have two tools to call get_current_time and get_current_date when you need to get the current time or date , when the user asks for the current time , you should call the get_current_time tool and when the user asks for the current date , you should call the get_current_date tool",
#     model = model,
#     tools=[get_current_time , get_current_date],
#     tool_use_behavior="run_llm_again"
    
# )

# When we tell the llm about the tools in the instructions but dont pass them to the Agent class then the llm outputs a make up tool code , e.g in our case the output of the llm is ```tool_code
# get_current_time()
# ```
# we can customized the behaviour of the tools using the tool_use_behavior and also by defining our own behaviour method

#  CASE 6 , What will happen when we set the tool_choice to required and dont pass the tools to the agent
# counter = 0
# class CustomHook(RunHooks):
#     async def on_agent_start(self, context:RunContextWrapper[Any], agent:Agent):
#         counter+1
#         print(f"Turn {counter} started.")
# run = CustomHook()        
# @function_tool
# def get_current_time():
#     print('Getting current time')
#     return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# @function_tool
# def get_current_date():
#     print('Getting current date')
#     return datetime.date.today().strftime("%Y-%m-%d")
# basic_agent= Agent(
#     name = "Basic Agent",
#     instructions="You are a basic agent that can answer questions. you have two tools to call get_current_time and get_current_date when you need to get the current time or date , when the user asks for the current time , you should call the get_current_time tool and when the user asks for the current date , you should call the get_current_date tool",
#     model = model,
#     tools=[get_current_date , get_current_time],
#     tool_use_behavior='run_llm_again',
#     # model_settings=ModelSettings(tool_choice="none" )
#     # reset_tool_choice=False,
    
# )

# When we dont pass the tools to the agent and set the tool_choice to required then the llm will not call any tool and will return am expected tool code in the output

# What if our query is not demanding to use the tool but we set the tool_choice to required ?

# A Bad Request Error is Received when we pass a query that is not demanding to use the tool but we set the tool_choice to required

# What if our query demands the tool calls but we have set the tool choice to none ?

# Then LLM provides the output that it has the tools but it does not call any tool , this is the response from the llm : ```tool_code
# today = get_current_date()
# current_time = get_current_time()
# print(f"Today's date is {today} and the current time is {current_time}")
# ```

# What if we set the reset_tool_choice to False?


# CASE 7 , if we set the tool_use_behaviour to stop at first tool and the tool output does not enforces the agent output type 

# class AgentRespose(BaseModel):
#     day: str
#     month : int 
#     year : int
# @function_tool
# def get_current_date():
#     print('Getting current date')
#     # return datetime.date.today().strftime("%Y-%m-%d")
#     response = AgentRespose(day=datetime.date.today().strftime("%Y-%m-%d") , month=datetime.date.today().month , year=datetime.date.today().year)
#     tool_output = response.model_dump(mode="json")
#     print('DEBUG' , tool_output)
#     return tool_output
# basic_agent= Agent(
#     name = "Basic Agent",
#     instructions="You are a basic agent that can answer questions. you have two tools to call get_current_time and get_current_date when you need to get the current time or date , when the user asks for the current time , you should call the get_current_time tool and when the user asks for the current date , you should call the get_current_date tool",
#     model = model,
#     tools=[get_current_date],
#     tool_use_behavior="run_llm_again",
#     # model_settings=ModelSettings(tool_choice="none" )
#     # output_type=AgentRespose
# )

# Case 8
#  Checking the max turns when a tool is called 
# @function_tool
# def get_weather(location: str) -> str:
#     '''
#     Get the weather for a given location

#     Args:
#         location (str): The location to get the weather for

#     Returns:
#         str: The weather for the location
#     '''
#     return f'''Weather in {location} is sunny '''

# basic_agent = Agent(
#     name = "Basic Agent",
#     instructions="You are a basic agent , that have a tool to get the weather of the given city",
#     model = model,
#     tools=[get_weather]
# )
#  A complete turn is a complete incovation of the llm , when a llm decides to call the tool , the Runner call the tool , at that time 1 turn is completed , and then the tool output with the input is passed to the llm and then the llm produces the output and then the turn is 2nd , So if the max turns is 1 and a  tool will be called then there will be a max turns exceeded error , 
# But if our llm does not initiate a tool call and can response on its own then max_turns =1 will be enough

#  CASE 8 , CREATING CUSTOM FUNCTION TOOL
# class weather_response(BaseModel):
#     city : str
#     country : str
#     current_season : str 
# @function_tool
# def get_weather(city : str):
#     '''
#       Get the weather for a given location

#       Args:
#           location (str): The location to get the weather for

#       Returns:
#           str: The weather for the location
#       '''
#     # return f'''Weather is {city} '''
#     raise ValueError("Custom Error")
    
  


# basic_agent = Agent(
#     name = "Basic Agent",
#     instructions="You are a basic agent , that have a tool to get the weather of the given city",
#     model = model,
#     tools=[get_weather]
   
# )

#  When an error is raised by the tool , it will not halt the Agent loop , this error is provided to the llm and then the llm provides the error message


#  CASE 10 : RUNRESULTS 
@function_tool
def get_weather(city : str):
    '''
      Get the weather for a given location

      Args:
          location (str): The location to get the weather for

      Returns:
          str: The weather for the location
      '''
    # return f'''Weather is {city} '''
    raise ValueError("Custom Error")
    
class weather_response(BaseModel):
    city : str 
    weather : str  


basic_agent = Agent(
    name = "Basic Agent",
    instructions="You are a basic agent , that have a tool to get the weather of the given city",
    model = model,
    tools=[get_weather],
    output_type=weather_response
   
)
#  We will validate the final output through final_ouptu_as method , and if the output is not valid then the method will raise an exception TypeError: Final output is not of type weather_response : 
# result.final_output_as(weather_response , raise_if_incorrect_type=True)


async def run_agent():
    result = await Runner.run(starting_agent=basic_agent, input="Hi ! tell the weather of the paris" , max_turns=5)
    print(result.final_output)
    # print(result.final_output_as(weather_response , raise_if_incorrect_type=True))
    print( result.to_input_list)
    

asyncio.run(run_agent())


