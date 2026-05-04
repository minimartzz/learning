"""
MCP Client
==========
Creating a Gradio-based MCP client using smolagents package
"""

import os

import gradio as gr
from dotenv import load_dotenv
from smolagents import CodeAgent, InferenceClientModel, MCPClient

load_dotenv()

try:
    # Connect to the MCP Server
    mcp_client = MCPClient(
        {
            "url": "https://abidlabs-mcp-tool-http.hf.space/gradio_api/mcp/sse",
            "transport": "sse",
        },
        structured_output=True,
    )
    tools = mcp_client.get_tools()
    print(
        "\n".join(f"{t.name}: {t.description}" for t in tools)
    )  # Tools available in server

    # Inference Client Model to get responses
    model = InferenceClientModel(token=os.environ.get("HF_TOKEN"))
    agent = CodeAgent(
        tools=[*tools],
        model=model,
        additional_authorized_imports=["json", "ast", "urllib", "base64"],
    )

    # Gradio UI
    demo = gr.ChatInterface(
        fn=lambda message, history: str(agent.run(message)),
        examples=["Analyze the sentiment of the following text 'This is awesome'"],
        title="Agent with MCP Tools",
        description="A simple agent that uses MCP tools to answer questions",
    )

    demo.launch(mcp_server=True)
finally:
    mcp_client.disconnect()
