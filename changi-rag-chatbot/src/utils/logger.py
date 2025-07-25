# src/utils/logger.py
import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict
from functools import lru_cache

try:
    from .config import settings
    LOG_LEVEL = settings.LOG_LEVEL
    LOG_FORMAT = settings.LOG_FORMAT
except ImportError:
    import os
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = os.getenv('LOG_FORMAT', 'json')

class JSONFormatter(logging.Formatter):
    """JSON log formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 'exc_text']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)

class SimpleLogger:
    """Simple console logger that's always available"""
    
    def __init__(self, name: str):
        self.name = name
    
    def info(self, msg, **kwargs): 
        print(f"ℹ️  [{self.name}] {msg}")
        if kwargs: 
            details = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
            print(f"   📝 {details}")
    
    def debug(self, msg, **kwargs): 
        print(f"🐛 [{self.name}] {msg}")
        if kwargs: 
            details = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
            print(f"   📝 {details}")
    
    def error(self, msg, **kwargs): 
        print(f"❌ [{self.name}] {msg}")
        if kwargs: 
            details = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
            print(f"   📝 {details}")
    
    def warning(self, msg, **kwargs): 
        print(f"⚠️  [{self.name}] {msg}")
        if kwargs: 
            details = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
            print(f"   📝 {details}")

@lru_cache()
def get_logger(name: str):
    """Get configured logger instance with fallback to simple logger"""
    try:
        # Try to create proper logger
        logger = logging.getLogger(name)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
        
        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        
        if LOG_FORMAT.lower() == "json":
            handler.setFormatter(JSONFormatter())
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger
        
    except Exception:
        # Fallback to simple logger
        return SimpleLogger(name)

# Export for backwards compatibility
def create_logger(name: str):
    """Create logger - backwards compatible function"""
    return get_logger(name)