"""
Lecture 2 — Mini-project: Support Ticket Classifier (Solution)

A complete reference implementation. Uses LangChain 1.0's init_chat_model +
with_structured_output() to classify free-text support tickets into a typed
TicketClassification object.

Run: `python 02-ticket-classifier.py`
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Allow `from llm_provider import ...` from anywhere in the repo.
# This file lives at course/part-XX-name/code/projects/solution/ — three levels
# deep from `course/`, so parents[3] points at the `course/` directory.
def _find_shared_path() -> str:
    """Walk up the directory tree until we find course/_shared/.

    Returns the absolute path to ``course/_shared/``. Works regardless of
    how deep the calling file is nested.
    """
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "_shared" / "llm_provider.py").is_file():
            return str(parent / "_shared")
    raise RuntimeError(
        "Could not locate course/_shared/llm_provider.py by walking up "
        f"from {here}. Check that the _shared directory exists."
    )


sys.path.insert(0, _find_shared_path())  # noqa: E402
from llm_provider import get_chat_model  # noqa: E402


# ─────────────────────────────────────────────────────────────
# Schema — what we want back from the model
# ─────────────────────────────────────────────────────────────
class TicketClassification(BaseModel):
    """A structured classification of a free-text support ticket."""

    category: str = Field(
        description="The category of the issue. One of: "
        "'billing', 'technical', 'account', 'general', 'logistics'."
    )
    priority: str = Field(
        description="How urgent the ticket is. One of: 'low', 'medium', 'high'."
    )
    sentiment: str = Field(
        description="The customer's emotional tone. One of: "
        "'positive', 'neutral', 'negative', 'angry'."
    )
    requires_human: bool = Field(
        description="True if this ticket should be escalated to a human agent."
    )


# ─────────────────────────────────────────────────────────────
# Classifier — chat model + structured output
# ─────────────────────────────────────────────────────────────
def build_classifier():
    """Build a callable that classifies ticket text → TicketClassification."""
    # `get_chat_model` dispatches to OpenAI (default) or MiniMax M3 based on
    # the `LLM_PROVIDER` env var. See course/_shared/README.md.
    # init_chat_model-style: it works with OpenAI, Anthropic, Google, etc.
    model = get_chat_model()

    # with_structured_output forces the model to return a valid Pydantic object.
    # No string parsing, no regex, no fragile JSON handling.
    return model.with_structured_output(TicketClassification)


# ─────────────────────────────────────────────────────────────
# Demonstration
# ─────────────────────────────────────────────────────────────
TEST_TICKETS = [
    "I keep getting charged for a subscription I cancelled last week. Very frustrating!",
    "My order hasn't arrived yet, and it's been over two weeks. Can you provide an update?",
    "Truck driver was rude and unhelpful when I called about my delivery. Not acceptable.",
]


def format_result(idx: int, ticket: str, result: TicketClassification) -> str:
    return (
        f"--- Ticket {idx} ---\n"
        f"  Text:       {ticket[:70]}{'...' if len(ticket) > 70 else ''}\n"
        f"  Category:   {result.category}\n"
        f"  Priority:   {result.priority}\n"
        f"  Sentiment:  {result.sentiment}\n"
        f"  Needs Human: {result.requires_human}"
    )


def main():
    load_dotenv()
    # Accept either OpenAI's key or MiniMax's key — provider is selected by
    # `LLM_PROVIDER` (defaults to 'openai'). See course/_shared/README.md.
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("MINIMAX_API_KEY")):
        raise RuntimeError(
            "No API key set. Add one of:\n"
            "  OPENAI_API_KEY=sk-...    (default — OpenAI)\n"
            "  MINIMAX_API_KEY=sk-...   (MiniMax M3)\n"
            "Optionally set LLM_PROVIDER=minimax to switch providers."
        )

    classifier = build_classifier()

    print("=" * 60)
    print("Support Ticket Classifier — Lecture 2 Solution")
    print("=" * 60)
    print(f"\nClassifying {len(TEST_TICKETS)} tickets...\n")

    for i, ticket in enumerate(TEST_TICKETS, 1):
        # Invoke the classifier — returns a typed Pydantic object directly
        result = classifier.invoke(ticket)
        print(format_result(i, ticket, result))
        print()

    # ── Bonus: show streaming doesn't apply to structured output,
    # but you CAN use the same prompt with a plain ChatModel for streaming.
    print("=" * 60)
    print("Why this matters")
    print("=" * 60)
    print(
        """
Without structured output, you'd be parsing strings like
    "category: billing, priority: high, sentiment: negative"
with regex and string splits. Fragile. Breaks the moment the model
adds an emoji or a different separator.

with_structured_output returns a Pydantic object — typed, validated,
no parsing. If the model can't satisfy the schema, the call fails
loudly instead of silently corrupting your data.
"""
    )


if __name__ == "__main__":
    main()
