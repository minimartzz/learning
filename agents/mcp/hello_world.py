from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Weather Service")

# Tool implementation
@mcp.tool()
def get_weather(location: str) -> str:
  """Get the current weatehr for a specific location"""
  return f"Weather in {location} is 72 degrees"

# Resource implementation
@mcp.resource("weather://{location}")
def weather_resource(location: str) -> str:
  """Provide weather data as a resource"""
  return f"Weather data for {location}: Sunny, 72 degrees"

# Prompt implementation
@mcp.prompt()
def weather_report(location: str) -> str:
  """Create a weather report prompt"""
  return f"You are a weather reporter. Weather report for {location}"

if __name__ == "__main__":
  mcp.run(transport="sse", port=3001)
