"""Root convenience entry point for Autonomous Academic Research Agent.
Delegates execution to research_agent.main.
"""

import sys
from pathlib import Path

# Add research_agent to sys.path
PROJECT_DIR = Path(__file__).resolve().parent / "research_agent"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from main import main

if __name__ == "__main__":
    main()
