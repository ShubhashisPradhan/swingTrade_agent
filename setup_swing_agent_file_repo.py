"""
setup_swing_agent_repo.py
--------------------------------
Creates the complete folder & file structure for the
Swing Trading AI Agent project (LangGraph + LangChain + Finnhub).
Run this script once in an empty folder:
    python setup_swing_agent_repo.py
"""

import os

# Define folder structure
folders = [
    "data/raw",
    "data/processed",
    "data/cache",
    "data/backtest_results",
    "data/logs",
    "notebooks",
    "src/agent/tools",
    "src/utils",
    "scripts",
    "ui/components",
    "tests"
]

# Define files to create (with optional boilerplate)
files = {
    # Agent core
    "src/agent/__init__.py": "",
    "src/agent/state.py": '"""Defines the SwingAgentState class for workflow."""\n\n',
    "src/agent/swing_agent_graph.py": '"""LangGraph workflow definition placeholder."""\n\n',
    "src/agent/llm_reasoning.py": '"""LangChain reasoning setup placeholder."""\n\n',
    
    # Tools
    "src/agent/tools/__init__.py": "",
    "src/agent/tools/data_tools.py": '"""Data fetching tools (Finnhub, Yahoo)."""',
    "src/agent/tools/indicator_tools.py": '"""Indicator computation tools (EMA, RSI, MACD)."""',
    "src/agent/tools/sentiment_tools.py": '"""Sentiment and news analysis tools."""',
    "src/agent/tools/signal_tools.py": '"""Signal generation tools (Pullback, Breakout)."""',
    "src/agent/tools/scoring_tools.py": '"""Stock scoring and ranking tools."""',
    "src/agent/tools/reporting_tools.py": '"""Reporting and visualization tools."""',
    
    # Utils
    "src/utils/__init__.py": "",
    "src/utils/logger.py": '"""Logging configuration."""',
    "src/utils/cache_manager.py": '"""Cache management utilities."""',
    "src/utils/config_loader.py": '"""YAML + .env config loader."""',
    
    # Scripts
    "scripts/run_swing_agent.py": '"""Entrypoint: run swing trading agent."""',
    "scripts/run_backtest.py": '"""Entrypoint: run backtests."""',
    "scripts/run_data_update.py": '"""Entrypoint: refresh market data cache."""',
    
    # UI
    "ui/streamlit_app.py": '"""Streamlit dashboard for daily insights."""',
    
    # Tests
    "tests/test_data_tools.py": '"""Unit tests for data tools."""',
    "tests/test_signal_tools.py": '"""Unit tests for signal tools."""',
    "tests/test_agent_workflow.py": '"""Integration tests for LangGraph workflow."""'
}

def create_project_structure():
    print("🚀 Creating Swing Trading Agent repo structure...\n")

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"📁 Created folder: {folder}")

    for file_path, content in files.items():
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
        print(f"📄 Created file: {file_path}")

    print("\n✅ Repository skeleton created successfully!")

if __name__ == "__main__":
    create_project_structure()
