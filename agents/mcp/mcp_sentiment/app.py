"""
Gradio MCP Server
=================
Gradio based MCP server to be deployed to HuggingFace Spaces
"""

import json

import gradio as gr
from textblob import TextBlob


def sentiment_analysis(text: str) -> str:
    """
    Analyze the sentiment of the provided text

    Args:
        text (str): Text to analyse sentiment

    Returns:
        str: JSON string containing polarity, subjectivity and assessment
    """
    blob = TextBlob(text)
    sentiment = blob.sentiment

    result = {
        "polarity": round(sentiment.polarity, 2),  # -1 to 1
        "subjectivity": round(sentiment.subjectivity, 2),  # 0 to 1
        "assessment": "positive"
        if sentiment.polarity > 0
        else "negative"
        if sentiment.polarity < 0
        else "neutral",
    }

    return json.dumps(result)


# Gradio Interface
demo = gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox(placeholder="Please enter your text..."),
    outputs=gr.Textbox(),
    title="Text Sentiment Analysis",
    description="Analyse the sentiment of the provided text using TextBlob",
)

if __name__ == "__main__":
    demo.launch(mcp_server=True)
