import sys
import os
import asyncio

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evals.eval_runner import main

if __name__ == "__main__":
    # Force UTF-8 encoding for Rich terminal output on Windows
    sys.stdout.reconfigure(encoding='utf-8')
    
    # Run the async eval pipeline
    asyncio.run(main())
