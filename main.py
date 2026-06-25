import uvicorn
import os
import logging

SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

from api import app

if __name__ == "__main__":
    logger.info("Starting Jina API on %s:%s", SERVER_HOST, SERVER_PORT)
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
