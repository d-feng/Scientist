"""
Utility functions for the E-Gen framework.
"""

import re
from typing import Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.utils.interactive_env import is_interactive_env
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

def get_llm(model_name: str, temperature: float = 0.0, port: Optional[int] = None, api_key: str = "EMPTY") -> BaseChatModel:
    """Get a language model instance.
    
    Args:
        model_name (str): Name of the model to use
        temperature (float): Temperature for generation
        port (Optional[int]): Port for local model server
        api_key (str): API key for hosted models
        
    Returns:
        BaseChatModel: Language model instance
    """
    if model_name == "cursor-claude":
        # Use Cursor's built-in Claude
        class CursorClaude(BaseChatModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                # This is a placeholder - Cursor will handle the actual model calls
                pass
                
            def _llm_type(self):
                return "cursor-claude"
                
        return CursorClaude()
    elif model_name.startswith("gpt"):
        return ChatOpenAI(
            model_name=model_name,  # Use the model name directly
            temperature=temperature,
            openai_api_key=api_key
        )
    elif model_name.startswith("claude"):
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            anthropic_api_key=api_key
        )
    else:
        raise ValueError(f"Model {model_name} not supported. Please use GPT, Claude, or cursor-claude models.")

def pretty_print(message: HumanMessage | AIMessage, printout: bool = True) -> str:
    """
    Pretty print a message.
    
    Args:
        message: Message to print
        printout: Whether to print to stdout
        
    Returns:
        str: Formatted message string
    """
    if isinstance(message, HumanMessage):
        title = "Human"
    else:
        title = "AI"
        
    divider = "=" * 32
    if is_interactive_env():
        formatted_title = f"\033[1m{title}\033[0m"
    else:
        formatted_title = title
        
    message_str = f"{divider}{formatted_title} Message{divider}\n{message.content}\n"
    
    if printout:
        print(message_str)
        
    return message_str 