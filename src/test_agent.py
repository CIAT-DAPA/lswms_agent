import asyncio
import os

from waterpoints_agent import WaterpointsAgent
from dotenv import load_dotenv

load_dotenv()
WATERPOINTS_MCP_URL = os.getenv("WATERPOINTS_MCP_URL", "https://mcp.waterpointsmonitoring.net/mcp")
WATERPOINTS_AGENT_MODEL = os.getenv("WATERPOINTS_AGENT_MODEL", "ollama/llama3.1:8b")
WATERPOINTS_AGENT_API_BASE = os.getenv("WATERPOINTS_AGENT_API_BASE", "http://192.168.199.91:11434")

async def main():
    agent = WaterpointsAgent(mcp_url=WATERPOINTS_MCP_URL, model=WATERPOINTS_AGENT_MODEL, api_base=WATERPOINTS_AGENT_API_BASE)

    response = await agent.chat(
        "Give me status of all waterpoints."
    )

    print(response)


if __name__ == "__main__":
    asyncio.run(main())