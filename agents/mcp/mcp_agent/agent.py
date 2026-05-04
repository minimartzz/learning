"""
MCP Agent
=========
Creating a Tiny Agent to connect to Gradio sentiment analysis server built earlier
"""

from huggingface_hub import Agent

agent = Agent(
    model="Qwen/Qwen2.5-72B-Instruct",
    provider="nebius",
    servers=[
        {
            "command": "npx",
            "args": ["mcp-remote", "http://localhost:7860/gradio_api/mcp/sse"],
        }
    ],
)
