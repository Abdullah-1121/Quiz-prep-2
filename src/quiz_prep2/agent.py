from datetime import datetime
from typing import Any, Dict, List
from agents import Agent, RunContextWrapper, Runner , AsyncOpenAI , OpenAIChatCompletionsModel , RunConfig, function_tool, set_tracing_disabled
import asyncio
from dotenv import load_dotenv
load_dotenv()
import os
set_tracing_disabled(True)
gemini_api_key = os.getenv("GEMINI_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")
# set_trace_processors([OpenAIAgentsTracingProcessor()])
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


# # Example 1: Basic string instructions
# agent_basic = Agent(
#     name="BasicAgent",
#     instructions="You are a helpful assistant that provides concise answers.",
#     model=model
# )

# # Example 2: Detailed string instructions
# agent_detailed = Agent(
#     name="DetailedAgent", 
#     instructions="""You are an expert Python developer.
#     - Always provide working code examples
#     - Explain your reasoning step by step
#     - Keep responses under 200 words
#     - Use best practices and modern Python syntax""",
#     model=model
# )

# # Example 3: Simple callable instructions
# def dynamic_instructions(context, agent):
#     print("RECEIVED CONTEXT", context)
#     print("RECEIVED AGENT", agent)
#     """Generate instructions based on context"""
#     return f"You are {agent.name}. Respond professionally and helpfully."

# agent_callable = Agent(
#     name="DynamicAgent",
#     instructions=dynamic_instructions,
#     model=model
# )

# # Example 4: Context-aware callable instructions
# def context_aware_instructions(context, agent):
#     print("RECEIVED CONTEXT", context)
#     print("RECEIVED AGENT", agent)
#     """Instructions that adapt based on conversation context"""
#     # You can access context.messages to see conversation history
#     message_count = len(getattr(context, 'messages', []))
    
#     if message_count == 0:
#         return "You are a friendly assistant. Introduce yourself and ask how you can help."
#     elif message_count < 3:
#         return "You are a helpful assistant. Be encouraging and detailed in your responses."
#     else:
#         return "You are an experienced assistant. Be concise but thorough."

# agent_context_aware = Agent(
#     name="ContextAwareAgent",
#     instructions=context_aware_instructions,
#     model=model
# )

# async def test_callable_instructions():
#     result1 = await Runner.run(agent_callable, "Hello!")
#     print("Callable Agent:", result1.final_output)
    
#     result2 = await Runner.run(agent_context_aware, "What's the weather like?")
#     print("Context Aware Agent:", result2.final_output)
    

# async def test_string_instructions():
#     result1 = await Runner.run(agent_basic, "What is Python?")
#     print("Basic Agent:", result1.final_output)
    
#     result2 = await Runner.run(agent_detailed, "How do I create a list comprehension?")
#     print("Detailed Agent:", result2.final_output)

# # Example 5: Async callable instructions
# async def async_instructions(context, agent):
#     """Async function that generates instructions"""
#     # Simulate async operation (like fetching from database)
#     await asyncio.sleep(0.1)
#     current_time = asyncio.get_event_loop().time()
#     parsed_time = datetime.fromtimestamp(current_time)
#     return f"""You are {agent.name}, an AI assistant with real-time capabilities.
#     Current timestamp: {parsed_time}
#     Provide helpful and timely responses."""

# agent_async = Agent(
#     name="AsyncAgent",
#     instructions=async_instructions,
#     model=model
# )

# async def test_async_instructions():
#     result = await Runner.run(agent_async, "What time is it?")
#     print("Async Agent:", result.final_output)
    
    
# from agents import Agent, Runner
# import asyncio

# # Example 7: Stateful callable instructions
# class InstructionGenerator:
#     def __init__(self):
#         self.interaction_count = 0
    
#     def __call__(self, context , agent):
#         self.interaction_count += 1
        
#         if self.interaction_count == 1:
#             return "You are a learning assistant. This is our first interaction - be welcoming!"
#         elif self.interaction_count <= 3:
#             return f"You are a learning assistant. This is interaction #{self.interaction_count} - build on our conversation."
#         else:
#             return f"You are an experienced learning assistant. We've had {self.interaction_count} interactions - be efficient."

# instruction_gen = InstructionGenerator()

# agent_stateful = Agent(
#     name="StatefulAgent",
#     instructions=instruction_gen,
#     model=model
# )

# async def test_stateful_instructions():
#     for i in range(4):
#         result = await Runner.run(agent_stateful, f"Question {i+1}: Tell me about Python")
#         print(f"Interaction {i+1}:", result.final_output[:100] + "...")
#         # print('Instructions No :', instruction_gen.interaction_count)


# def main():
#     # print("\n[1. STRING INSTRUCTIONS]")
#     # # # Test this
#     # asyncio.run(test_string_instructions())
    
#     # print("\n[2. CALLABLE INSTRUCTIONS]")
#     # asyncio.run(test_callable_instructions())
    
#     # print("\n[3. ASYNC INSTRUCTIONS]")
#     # asyncio.run(test_async_instructions())
    
#     print("\n[4. STATEFUL INSTRUCTIONS]")
#     asyncio.run(test_stateful_instructions())
    
    
# if __name__ == "__main__":
#     main()

# 
# class SessionContext:
#     def __init__(self):
#         self.messages_count: int = 0
#         self.topics_discussed: set[str] = set()
#         self.user_mood: str = "neutral"
#         self.start_time: datetime = datetime.now()

#     def add_message(self):
#         self.messages_count += 1

#     def add_topic(self, topic: str):
#         self.topics_discussed.add(topic)

#     def set_mood(self, mood: str):
#         self.user_mood = mood
        
#     def get_session_info(self) -> str:
#         return f"Messages: {self.messages_count}\nTopics: {self.topics_discussed}\nMood: {self.user_mood}"

# # Tools that work with context passed to Runner


# @function_tool
# def track_topic(ctx: RunContextWrapper[SessionContext], topic: str) -> str:
#     """Track a discussion topic"""
#     ctx.context.add_topic(topic)
#     return f"Tracking topic: {topic}"


# @function_tool
# def set_mood(ctx: RunContextWrapper[SessionContext], mood: str) -> str:
#     """Set the user's current mood"""
#     ctx.context.set_mood(mood)
#     return f"User mood set to: {mood}"


# @function_tool
# def get_session_info(ctx: RunContextWrapper[SessionContext]) -> str:
#     """Get current session information"""
#     return ctx.context.get_session_info()


# # Simple agent without dynamic instructions
# session_agent = Agent(
#     name="SessionAgent",
#     instructions="Help users track topics and mood. Be helpful and engaging. Always use the tools to update the context.",
#     tools=[track_topic, set_mood, get_session_info],
#     model = model
# )


# async def test_session_agent():
#     print("\n=== Testing Session Agent ===")

#     # Create context
#     context = SessionContext()

#     interactions = [
#         "I want to learn about Python programming",
#         "I'm getting frustrated with coding",
#         "Can you help me with data structures?",
#         "What's my current session info?"
#     ]

#     for i, message in enumerate(interactions):
#         context.add_message()

#         print(f"\n--- Interaction {i+1} ---")
#         print(f"Messages count: {context.messages_count}")
#         print(f"User message: {message}")

#         # Run the agent (context is available but tools work independently)
#         result = await Runner.run(session_agent, message, context=context)

#         print(f"Agent response: {result.final_output}")


#     print(f"\nFinal session state:")
#     print(f"Messages: {context.messages_count}")
#     print(f"Topics: {context.topics_discussed}")
#     print(f"Mood: {context.user_mood}")

# asyncio.run(test_session_agent())

class FitnessContext:
    def __init__(self):
        self.user_profile: Dict[str, Any] = {"name": None, "goal": None, "experience": "beginner"}
        self.workout_log: List[str] = []
        self.preferences: Dict[str, str] = {}
        self.injuries: List[str] = []

    def log_workout(self, workout: str):
        self.workout_log.append(workout)
        if len(self.workout_log) > 10:
            self.workout_log.pop(0)

    def add_injury(self, injury: str):
        if injury not in self.injuries:
            self.injuries.append(injury)


@function_tool
def log_workout(ctx: RunContextWrapper[FitnessContext], workout: str) -> str:
    ctx.context.log_workout(workout)
    return f"Workout logged: {workout}"

@function_tool
def update_goal(ctx: RunContextWrapper[FitnessContext], goal: str) -> str:
    ctx.context.user_profile["goal"] = goal
    return f"Updated goal to: {goal}"

@function_tool
def update_name(ctx: RunContextWrapper[FitnessContext], name: str) -> str:
    ctx.context.user_profile["name"] = name
    return f"Hi {name}, I've saved your name!"

@function_tool
def add_injury(ctx: RunContextWrapper[FitnessContext], injury: str) -> str:
    ctx.context.add_injury(injury)
    return f"Injury noted: {injury}"

@function_tool
def show_status(ctx: RunContextWrapper[FitnessContext]) -> Dict[str, Any]:
    return {
        "profile": ctx.context.user_profile,
        "workouts": ctx.context.workout_log,
        "injuries": ctx.context.injuries
    }


fitness_agent = Agent(
    name="FitnessCoach",
    instructions="""You are a personal fitness coach. 
- Greet the user and remember their name.
- Track workouts, goals, and injuries.
- Suggest personalized advice over time.
Use your tools to log and recall this data.""",
    tools=[log_workout, update_goal, update_name, add_injury, show_status],
    model=model
)


async def test_fitness_agent():
    context = FitnessContext()
    messages = [
        "Hi, I'm Sara and my goal is to lose fat",
        "I just did a 30-minute HIIT session",
        "Log an injury: left knee pain",
        "I completed a yoga workout today",
        "What's my fitness status?"
    ]
    
    for i, msg in enumerate(messages):
        print(f"\n--- User Message {i+1} ---")
        print("User:", msg)
        result = await Runner.run(fitness_agent, msg, context=context)
        print("Agent:", result.final_output)

        # Optional debug:
        print("Status Snapshot:", {
            "Profile": context.user_profile,
            "Workouts": context.workout_log,
            "Injuries": context.injuries
        })


asyncio.run(test_fitness_agent())