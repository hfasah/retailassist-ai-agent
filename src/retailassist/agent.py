"""Interactive command-line chat with the RetailAssist agent.

    retailassist                    # start a new conversation
    retailassist --session <id>     # pick up a saved conversation
"""

import argparse
import uuid

from .graph import ProviderMismatchError, RetailAssistant
from .llm import LLMAuthError, LLMRateLimitError, LLMUnavailableError


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with RetailAssist.")
    parser.add_argument("--session", help="ID of a saved conversation to continue")
    args = parser.parse_args()

    assistant = RetailAssistant()
    if args.session and not assistant.has_conversation(args.session):
        parser.error(f"No saved conversation with ID {args.session}.")
    session_id = args.session or uuid.uuid4().hex
    print(f"RetailAssist is ready ({assistant.llm.name}: {assistant.llm.model}, session {session_id}). "
          "Type 'quit' to exit.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_text.lower() in {"quit", "exit"}:
            break
        if not user_text:
            continue
        try:
            result = assistant.ask(session_id, user_text)
        except LLMAuthError as e:
            print(f"Error: {e}")
            break
        except ProviderMismatchError as e:
            print(f"Error: {e} Start a new conversation without --session.")
            break
        except LLMRateLimitError:
            print("Assistant: I'm getting a lot of requests right now. Please try again in a moment.\n")
            continue
        except LLMUnavailableError:
            print("Assistant: I couldn't reach the model. Check your internet connection and try again.\n")
            continue
        print(f"Assistant: {result['reply']}")
        for citation in result["citations"]:
            print(f"  [{citation['number']}] {citation['title']} ({citation['source']})")
        print()

    print(f"To continue this conversation later: retailassist --session {session_id}")


if __name__ == "__main__":
    main()
