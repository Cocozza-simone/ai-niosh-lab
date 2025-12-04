"""
Retry logic with exponential backoff for Ollama API calls.
Handles timeouts, connection errors, and occasional glitches automatically.
"""

import time
import random
import asyncio
from typing import Callable, Any, Optional, Union, List
from functools import wraps
import ollama
from rich.console import Console
import sys
import os
import io

# Configure console for Windows UTF-8 support
if sys.platform == "win32":
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    try:
        import locale
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        pass
    try:
        os.system('chcp 65001 > nul 2>&1')
    except:
        pass

# Create a custom stdout with UTF-8 encoding for Windows
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

console = Console(
    force_terminal=True,
    legacy_windows=False,
    emoji=False,  # Disable emoji rendering to avoid Unicode issues
    markup=True
)


class RetryConfig:
    """Configuration for retry logic"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on_exceptions: tuple = (
            ConnectionError,
            TimeoutError,
            Exception,  # Generic exception for ollama API issues
        )
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_on_exceptions = retry_on_exceptions


class RetryError(Exception):
    """Raised when all retry attempts are exhausted"""
    
    def __init__(self, message: str, attempts: int, original_exception: Exception):
        self.message = message
        self.attempts = attempts
        self.original_exception = original_exception
        super().__init__(f"Failed after {attempts} attempts: {message}")


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for exponential backoff with optional jitter"""
    
    delay = min(
        config.base_delay * (config.exponential_base ** attempt),
        config.max_delay
    )
    
    if config.jitter:
        # Add random jitter between 0-25% of the delay
        jitter_amount = delay * 0.25 * random.random()
        delay += jitter_amount
    
    return delay


def retry_sync(config: Optional[RetryConfig] = None):
    """Decorator for retrying synchronous functions"""
    
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    if attempt > 0:
                        delay = calculate_delay(attempt - 1, config)
                        console.print(f"[yellow]WARNING: Retry attempt {attempt}/{config.max_retries} after {delay:.1f}s delay...[/]")
                        time.sleep(delay)
                    
                    return func(*args, **kwargs)
                    
                except config.retry_on_exceptions as e:
                    last_exception = e
                    if attempt == config.max_retries:
                        console.print(f"[red]ERROR: All {config.max_retries + 1} attempts failed[/]")
                        break
                    
                    console.print(f"[yellow]WARNING: Attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:100]}...[/]")
                    
                except Exception as e:
                    # Don't retry on exceptions that aren't in the retry list
                    console.print(f"[red]ERROR: Non-retryable error: {type(e).__name__}: {str(e)[:100]}...[/]")
                    raise
            
            # All attempts failed
            raise RetryError(
                f"Function {func.__name__} failed after {config.max_retries + 1} attempts",
                config.max_retries + 1,
                last_exception
            )
        
        return wrapper
    return decorator


def retry_async(config: Optional[RetryConfig] = None):
    """Decorator for retrying asynchronous functions"""
    
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    if attempt > 0:
                        delay = calculate_delay(attempt - 1, config)
                        console.print(f"[yellow]WARNING: Retry attempt {attempt}/{config.max_retries} after {delay:.1f}s delay...[/]")
                        await asyncio.sleep(delay)
                    
                    return await func(*args, **kwargs)
                    
                except config.retry_on_exceptions as e:
                    last_exception = e
                    if attempt == config.max_retries:
                        console.print(f"[red]ERROR: All {config.max_retries + 1} attempts failed[/]")
                        break
                    
                    console.print(f"[yellow]WARNING: Attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:100]}...[/]")
                    
                except Exception as e:
                    # Don't retry on exceptions that aren't in the retry list
                    console.print(f"[red]ERROR: Non-retryable error: {type(e).__name__}: {str(e)[:100]}...[/]")
                    raise
            
            # All attempts failed
            raise RetryError(
                f"Function {func.__name__} failed after {config.max_retries + 1} attempts",
                config.max_retries + 1,
                last_exception
            )
        
        return wrapper
    return decorator


# Enhanced Ollama client with retry logic
class RobustOllamaClient:
    """Enhanced Ollama client with automatic retry logic"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
    
    @retry_sync()
    def chat(self, model: str, messages: List[dict], **kwargs) -> dict:
        """Chat with retry logic"""
        try:
            return ollama.chat(model=model, messages=messages, **kwargs)
        except Exception as e:
            console.print(f"[red]ERROR: Ollama chat error: {type(e).__name__}: {str(e)[:100]}...[/]")
            raise
    
    @retry_sync()
    def generate(self, model: str, prompt: str, **kwargs) -> dict:
        """Generate with retry logic"""
        try:
            return ollama.generate(model=model, prompt=prompt, **kwargs)
        except Exception as e:
            console.print(f"[red]ERROR: Ollama generate error: {type(e).__name__}: {str(e)[:100]}...[/]")
            raise
    
    @retry_sync()
    def embed(self, model: str, input: str, **kwargs) -> dict:
        """Embed with retry logic"""
        try:
            return ollama.embed(model=model, input=input, **kwargs)
        except Exception as e:
            console.print(f"[red]ERROR: Ollama embed error: {type(e).__name__}: {str(e)[:100]}...[/]")
            raise


# Context managers for with-statement usage
class RetryContext:
    """Context manager for retry operations"""
    
    def __init__(self, config: Optional[RetryConfig] = None, operation_name: str = "Operation"):
        self.config = config or RetryConfig()
        self.operation_name = operation_name
        self.attempts = 0
    
    def __enter__(self):
        self.attempts = 0
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if issubclass(exc_type, self.config.retry_on_exceptions):
                # This will be caught by the retry logic
                return False  # Don't suppress the exception
        return False


def execute_with_retry(func: Callable, *args, config: Optional[RetryConfig] = None, operation_name: str = "Operation", **kwargs) -> Any:
    """Execute a function with retry logic using decorators"""
    
    @retry_sync(config)
    def _execute():
        return func(*args, **kwargs)
    
    console.print(f"[blue]🔄 Executing {operation_name} with retry logic...[/]")
    return _execute()


async def execute_with_retry_async(func: Callable, *args, config: Optional[RetryConfig] = None, operation_name: str = "Operation", **kwargs) -> Any:
    """Execute an async function with retry logic using decorators"""
    
    @retry_async(config)
    async def _execute():
        return await func(*args, **kwargs)
    
    console.print(f"[blue]🔄 Executing {operation_name} with retry logic...[/]")
    return await _execute()


# Specialized retry configurations for different scenarios
def get_fast_retry_config() -> RetryConfig:
    """Fast retry config for quick operations"""
    return RetryConfig(
        max_retries=2,
        base_delay=0.5,
        max_delay=5.0,
        exponential_base=1.5,
        jitter=True
    )


def get_slow_retry_config() -> RetryConfig:
    """Slow retry config for long operations"""
    return RetryConfig(
        max_retries=5,
        base_delay=2.0,
        max_delay=120.0,
        exponential_base=2.5,
        jitter=True
    )


def get_critical_retry_config() -> RetryConfig:
    """Critical retry config for important operations"""
    return RetryConfig(
        max_retries=10,
        base_delay=1.0,
        max_delay=300.0,  # 5 minutes max
        exponential_base=2.0,
        jitter=True
    )


# Usage examples and utilities
def demonstrate_retry_usage():
    """Demonstrate how to use the retry logic"""
    
    # Example 1: Using decorators
    @retry_sync()
    def risky_operation():
        """Example function that might fail"""
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise ConnectionError("Random connection error")
        return "Success!"
    
    # Example 2: Using the robust client
    client = RobustOllamaClient()
    
    # Example 3: Using context manager
    try:
        with RetryContext(operation_name="AI Generation"):
            result = risky_operation()
    except RetryError as e:
        console.print(f"[red]Operation failed completely: {e}[/]")


# Health check for Ollama
def check_ollama_health(model: str = "gemma3:12b") -> bool:
    """Check if Ollama is healthy and model is available"""
    
    try:
        # Try a simple generation to test health (without timeout parameter)
        result = ollama.generate(
            model=model,
            prompt="Hello",
            options={"temperature": 0.1, "num_predict": 5}
        )
        return True
    except Exception as e:
        console.print(f"[red]Ollama health check failed: {type(e).__name__}: {str(e)[:100]}...[/]")
        return False


# Batch retry for multiple operations
async def batch_retry_with_backoff(
    operations: List[tuple],  # List of (func, args, kwargs) tuples
    config: Optional[RetryConfig] = None,
    concurrent_limit: int = 3
) -> List[Any]:
    """Execute multiple operations with retry logic and concurrency control"""
    
    if config is None:
        config = RetryConfig()
    
    semaphore = asyncio.Semaphore(concurrent_limit)
    
    async def execute_with_semaphore(func, args, kwargs):
        async with semaphore:
            return await execute_with_retry_async(func, *args, config=config, **kwargs)
    
    console.print(f"[blue]🔄 Executing {len(operations)} operations with retry logic (concurrency: {concurrent_limit})...[/]")
    
    tasks = [
        execute_with_semaphore(func, args, kwargs)
        for func, args, kwargs in operations
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    success_count = sum(1 for r in results if not isinstance(r, Exception))
    console.print(f"[green]SUCCESS: Batch completed: {success_count}/{len(operations)} successful[/]")
    
    return results


if __name__ == "__main__":
    # Test the retry logic
    console.print("[info]Testing retry logic...[/]")
    
    # Test health check
    if check_ollama_health():
        console.print("[green]SUCCESS: Ollama is healthy[/]")
    else:
        console.print("[red]ERROR: Ollama health check failed[/]")
    
    # Demonstrate retry usage
    demonstrate_retry_usage()