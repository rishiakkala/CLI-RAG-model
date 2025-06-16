import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

class BaseAgent:
    """Base class for all agents in the system."""
    
    def __init__(self, agent_type: str, config: Optional[Dict[str, Any]] = None):
        self.agent_type = agent_type
        self.config = config or {}
        
        # Setup logging
        self.logger = logging.getLogger(f"agent.{agent_type}")
        
        # Activity log
        self.activity_log = []
    
    def log_activity(self, message: str, level: str = "info") -> None:
        """Log agent activity with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "message": message,
            "level": level
        }
        
        self.activity_log.append(log_entry)
        
        # Also log to the logger
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
    
    def get_activity_log(self) -> list:
        """Return the agent's activity log."""
        return self.activity_log
    
    def process(self, *args, **kwargs):
        """Process method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the process method")