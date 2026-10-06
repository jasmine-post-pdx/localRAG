"""Shared Rich console for the local RAG system's terminal output.

Every script prints through ``console`` so its messages look the same and
stand out from library output such as LangChain warnings.

Styles:
    info: Status lines (cyan).
    success: Completed steps (green).
    question: Prompts that wait for the user to type (bold yellow).
    warning: Problems the user should act on but that don't stop the run
        (magenta).
"""
from rich.console import Console
from rich.theme import Theme

# Markup is off so square brackets in messages (e.g. "[y/N]", file names)
# print literally instead of being parsed as Rich style tags.
console = Console(
    theme=Theme({
        "info": "cyan",
        "success": "green",
        "question": "bold yellow",
        "warning": "magenta",
    }),
    markup=False,
    highlight=False,
)
