from playwright_mcp import PlaywrightMCPClient
from litellm import completion
from utils import parse_tools

import asyncio
from dotenv import load_dotenv

import json
from prompts import DEFAULT_SYSTEM_PROMPT

from colorama import Fore, Style

load_dotenv()

class BrowserAgent:
    def __init__(
        self,
        model_name: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        
        self.mcp_client = PlaywrightMCPClient()
        self.model_name = model_name
        self.messages = [{"role": "system", "content": system_prompt}]
    
    async def initialize(self):
        """Initialize the agent asynchronously"""
        tools = await self.mcp_client.connect_to_playwright()
        self.formatted_tools = parse_tools(tools)
        return self
    
    def chat_completion(self):

        response = completion(
            model=self.model_name,
            messages=self.messages,
            tools=self.formatted_tools,
            tool_choice="auto")
        
        return response.choices[0].message
    
    async def agent_loop(self):
        while True:
            response_message = self.chat_completion()
            self.messages.append(response_message)

            if not response_message.tool_calls:
                print("No tool calls here. Terminating loop...")
                self.messages.append({"role": "assistant", "content": response_message.content})
                return response_message.content
            else:
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    print(f"Calling tool {function_name} with args {function_args}")

                    tool_result = await self.mcp_client.execute_tool(
                        tool_name=function_name,
                        tool_args=function_args)
                    
                    # Add the tool response to the conversation
                    self.messages.append(
                        {
                        "tool_call_id": tool_call.id, 
                        "role": "tool",
                        "name": function_name,
                        "content": tool_result
                        })

    async def process_query(self, query: str) -> str:
        self.messages.append({"role": "user", "content": query})
        final_response = await self.agent_loop()

        return final_response
    
async def main():
    agent = BrowserAgent(model_name="gemini/gemini-2.0-flash")
    
    await agent.initialize()

    try:
        query = input(Fore.GREEN + "What would you like me to do?: " + Style.RESET_ALL)
        response = await agent.process_query(query)
        
        print(Fore.BLUE + "Assistant: " + Style.RESET_ALL + response)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await agent.mcp_client.cleanup()

if __name__ == '__main__':
    asyncio.run(main())