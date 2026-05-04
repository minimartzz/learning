# PR Agent Workflow Server

An MCP server that demonstrates how to make team-aware and workflow-intelligence

## 🎯 GOAL

Automatically recommend PR layouts, monitors Github Action checks and notifies team members

- Suggets the right PR template based on changed files
- Monitor GitHub Actions runs and provide formatted summaries
- Automatically notify team via Slack when deployments fail/ succeed
- Guide developers thorugh team-specific review processes based on Actions results

## 🔨 TOOLS

- Tools and Prompt templates
- Cloudflare Tunnel to receive webhooks and process CI/CD events
- HuggingFace main LLM and agents provider
- GitHub
- Slack
- Claude Code

## 🤖 MCP Server

Provide Claude with raw git data and allow it to suggest PR templates based on the provided info

- **Flexible Analysis:** Claude understands the context that simples rules will miss
- **Natural Language:** Suggestions are human, not robotic
- **Adaptable:** Works for any codebase or coding style

Components

1. `analyse_file_changes` - Retrive git diff information and changed files (raw data)
2. `get_pr_templates` - List available PR templates (resource management)
3. `suggest_tempalte` - Allows Claude to recommend the most appropriate template (decision-making)
