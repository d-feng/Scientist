"""
Configuration settings for E-Gen framework.
"""

import os
from pathlib import Path

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'EMPTY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', 'EMPTY')

# Create .env file if it doesn't exist
def create_env_file():
    """Create a .env file with template API keys if it doesn't exist."""
    env_path = Path(__file__).parent.parent / '.env'
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write("""# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
""") 