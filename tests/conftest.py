"""
Shared fixtures and path setup for rag-aya tests.
"""

import sys
import os

# Add project root to sys.path so that `import chunker` etc. work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
