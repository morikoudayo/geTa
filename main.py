"""
llama.geta: a Gateway Enabling Tool Access

Main entry point for the application.
"""

from geta.logging_config import setup_logging
from geta import agent
from geta.api import create_app

# Setup logging
logger = setup_logging()

# Initialize agent before creating app
try:
    agent.initialize()
except Exception as e:
    logger.error(f"Failed to initialize agent: {e}")
    logger.info("Please check your llama.cpp server configuration")
    raise

# Create FastAPI app (agent must be initialized first)
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
