import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from typing import Any

class PlaywrightMCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
    
    async def connect_to_playwright(self):
        """Connect to the Playwright MCP server
        
        Args:
        server_script_path: Path to the server script (.py or .js)
        """
        
        server_params = StdioServerParameters(
            command="npx", 
            args=["@playwright/mcp@latest",
            "--browser=chrome"])

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        
        print("Connected to Playwright Server!")

        return tools
    
    async def execute_tool(self, tool_name: str, tool_args: dict[str, Any]):
        """Call a tool"""
        response = await self.session.call_tool(tool_name, tool_args)
        return response.content[0].text
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.exit_stack:
            await self.exit_stack.aclose()
    
async def main():
    client = PlaywrightMCPClient()
    try:
        tools = await client.connect_to_playwright()
        print("\n")
        print("TOOLS AVAILABLE")
        print('\n')
        for tool in tools:
            print(tool)
            print('\n')
    finally:
        await client.cleanup()
        print("Clean Up Successful!")

if __name__ == "__main__":
    asyncio.run(main())
    