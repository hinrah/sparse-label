import logging

# Configure the logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a logger instance
logger = logging.getLogger(__name__)