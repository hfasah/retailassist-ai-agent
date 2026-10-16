"""Interactive command-line chat with the RetailAssist agent.

    retailassist                    # start a new conversation
    retailassist --session <id>     # pick up a saved conversation
"""

import argparse
import uuid

from .graph import ApprovalPendingError, ProviderMismatchError, RetailAssistant
from .llm import LLMAuthError, LLMError, LLMRateLimitError, LLMUnavailableError
from .observability import get_observability


def show(result: dict) -> None:
    print(f"Assistant: {result['reply']}")
    for citation in result["citations"]:
        print(f"  [{citation['number']}] {citation['title']} ({citation['source']})")
    for request in result["pending_approvals"]:
        print(f"  Waiting for approval: {request['request_id']} ({request['title']}). Staff can decide with "
              f"POST /approvals/{int(request['request_id'].removeprefix('APR-'))}; press Enter to check.")
    print()


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

    waiting = bool(assistant.pending_approvals(session_id))
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_text.lower() in {"quit", "exit"}:
            break
        if waiting:  # paused for approval: see whether staff have decided
            try:
                result = assistant.resume(session_id) if not assistant.pending_approvals(session_id) \
                    else assistant.result(session_id)
            except LLMError as e:
                print(f"Assistant: I couldn't continue just yet ({e}). Press Enter to try again.\n")
                continue
            if result["pending_approvals"]:
                print("Assistant: Still waiting for a team member. Press Enter to check again.\n")
                continue
            waiting = False
            show(result)
            continue
        if not user_text:
            continue
        try:
            result = assistant.ask(session_id, user_text)
        except ApprovalPendingError as e:
            print(f"Assistant: {e}.\n")
            waiting = True
            continue
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
        show(result)
        waiting = bool(result["pending_approvals"])

    get_observability().shutdown()  # send any traces still buffered
    print(f"To continue this conversation later: retailassist --session {session_id}")


if __name__ == "__main__":
    main()
