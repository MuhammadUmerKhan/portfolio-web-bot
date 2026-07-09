import time
import asyncio
from functools import wraps
from typing import Callable, Any
import logfire

class CircuitBreakerOpenException(Exception):
    pass

class AsyncCircuitBreaker:
    """
    An asynchronous circuit breaker that tracks consecutive failures.
    If failures exceed the threshold, the circuit trips OPEN, instantly failing 
    subsequent calls until the recovery timeout expires, at which point it 
    transitions to HALF_OPEN to test if the service has recovered.
    """
    def __init__(self, name: str, failure_threshold: int = 3, recovery_timeout: float = 30.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def _check_state(self):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logfire.info(f"🔄 Circuit breaker '{self.name}' transitioning to HALF_OPEN")
            else:
                logfire.warning(f"🛑 Circuit breaker '{self.name}' is OPEN. Skipping call.")
                raise CircuitBreakerOpenException(f"Circuit breaker '{self.name}' is OPEN")

    def _record_success(self):
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            logfire.info(f"✅ Circuit breaker '{self.name}' recovered and is now CLOSED")

    def _record_failure(self, e: Exception):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logfire.error(f"🛑 Circuit breaker '{self.name}' tripped OPEN after {self.failure_count} failures. Error: {e}")

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            self._check_state()
            try:
                result = await func(*args, **kwargs)
                self._record_success()
                return result
            except Exception as e:
                self._record_failure(e)
                raise e

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            self._check_state()
            try:
                result = func(*args, **kwargs)
                self._record_success()
                return result
            except Exception as e:
                self._record_failure(e)
                raise e

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
