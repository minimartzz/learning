"""
Gradio MCP Server
=================
Gradio offers a straightforward way to expose AI model capcabilities
through the standardized API protocol. Create AI-accessible tools with
minimal code so share MCP server with others

Note on demo.launch(mcp_server=True)
    - Gradio functions are automatically converted to MCP Tools
    - Gradio server now also listens for MCP protocol messages
    - “View API” link in the footer of your Gradio app, and then click on “MCP” to view
        the endpoint
"""

import gradio as gr


def letter_counter(word: str, letter: str) -> int:
    """
    Count the number of occurrences of a letter in a word or text

    Args:
        word (str): The input text to search through
        letter (str): The letter to search for

    Returns:
        int: The number of times the letter appears in the text
    """
    word = word.lower()
    letter = letter.lower()
    count = word.count(letter)

    return count


demo = gr.Interface(
    fn=letter_counter,
    inputs=["textbox", "textbox"],
    outputs="number",
    title="Letter Counter",
    description="Enter text and a letter to coutn how many times the letter appears in"
    "the text",
)

if __name__ == "__main__":
    demo.launch(mcp_server=True)
