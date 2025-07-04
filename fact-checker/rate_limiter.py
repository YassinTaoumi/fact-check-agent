"""
Rate limiter utility for controlling API calls.
"""

import asyncio
import time
import logging
from typing import Callable, Any
from functools import wraps
from config import config

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_calls_per_minute: int = 30):
        self.max_calls_per_minute = max_calls_per_minute
        self.calls = []
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        async with self.lock:
            now = time.time()
            
            # Remove calls older than 1 minute
            self.calls = [call_time for call_time in self.calls if now - call_time < 60]
            
            # Check if we need to wait
            if len(self.calls) >= self.max_calls_per_minute:
                # Find the oldest call within the minute
                oldest_call = min(self.calls)
                wait_time = 60 - (now - oldest_call) + 2  # Add 2 second buffer
                
                if wait_time > 0:
                    logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time)
                    
                    # Update now after waiting
                    now = time.time()
                    self.calls = [call_time for call_time in self.calls if now - call_time < 60]
            
            # Add minimum delay between calls (3 seconds)
            if self.calls:
                last_call = max(self.calls)
                time_since_last = now - last_call
                min_delay = 3.0
                if time_since_last < min_delay:
                    wait_time = min_delay - time_since_last
                    logger.info(f"Adding minimum delay: {wait_time:.1f} seconds")
                    await asyncio.sleep(wait_time)
                    now = time.time()
            
            # Record this call
            self.calls.append(now)

# Global rate limiter instance
llm_rate_limiter = RateLimiter(config.requests_per_minute)

def rate_limited_llm_call(func: Callable) -> Callable:
    """Decorator to rate limit LLM calls."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        await llm_rate_limiter.wait_if_needed()
        return func(*args, **kwargs)
    
    return wrapper
