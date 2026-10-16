"""Interactive command-line chat with the RetailAssist agent.

    retailassist                    # start a new conversation
    retailassist --session <id>     # pick up a saved conversation
"""

import argparse
import uuid

import anthropic

from .graph import RetailAssistant


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with RetailAssist.")
    parser.add_argument("--session", help="ID of a saved conversation to continue")
    args = parser.parse_args()

    assistant = RetailAssistant()
    if args.session and not assistant.has_conversation(args.session):
        parser.error(f"No saved conversation with ID {args.session}.")
    session_id = args.session or uuid.uuid4().hex
    print(f"RetailAssist is ready (session {session_id}). Type 'quit' to exit.\n")

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
        except (anthropic.AuthenticationError, anthropic.CredentialsError):
            print("Error: invalid or missing API key. Set ANTHROPIC_API_KEY in your environment.")
            break
        except TypeError as e:
            # The SDK raises TypeError before sending anything when no credentials are configured.
            if "authentication method" not in str(e):
                raise
            print("Error: no API key found. Set ANTHROPIC_API_KEY in your environment.")
            break
        except anthropic.RateLimitError:
            print("Assistant: I'm getting a lot of requests right now. Please try again in a moment.\n")
            continue
        except anthropic.APIConnectionError:
            print("Assistant: I couldn't reach the server. Check your internet connection.\n")
            continue
        except anthropic.APIStatusError as e:
            print(f"Assistant: Something went wrong (HTTP {e.status_code}). Please try again.\n")
            continue
        print(f"Assistant: {result['reply']}")
        for citation in result["citations"]:
            print(f"  [{citation['number']}] {citation['title']} ({citation['source']})")
        print()

    print(f"To continue this conversation later: retailassist --session {session_id}")


if __name__ == "__main__":
    main()
