"""Interactive command-line chat with the RetailAssist agent."""

import anthropic

from .tools import ALL_TOOLS

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You are RetailAssist, a friendly shopping and support assistant for an outdoor gear store.

Help customers find products, check prices and stock, track their orders, and understand the store's \
return and shipping policies. Use your tools rather than guessing; if a tool finds nothing, say so. Base \
answers about returns, refunds, and shipping on the policy text, and use the order's dates and status to \
say how the policy applies to that customer. To look up an order, ask for the order number and the \
email it was placed with. Keep replies short and conversational.

You can't start returns, issue refunds, cancel orders, or change addresses yourself. When a customer \
wants one of those, tell them whether the policy allows it and that a team member will follow up."""


class RetailAssistant:
    def __init__(self, client: anthropic.Anthropic | None = None):
        self.client = client or anthropic.Anthropic()
        self.messages: list = []

    def ask(self, user_text: str) -> str:
        """Send one customer message, let Claude call tools as needed, and return its reply."""
        self.messages.append({"role": "user", "content": user_text})
        runner = self.client.beta.messages.tool_runner(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=ALL_TOOLS,
            messages=self.messages,
            output_config={"effort": "medium"},
            # If a safety classifier declines, retry on Anthropic's recommended fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

        final = None
        for message in runner:
            # Keep the full turn (including tool calls and results) so the history stays append-only.
            self.messages.append(message.to_param())
            tool_results = runner.generate_tool_call_response()
            if tool_results is not None:
                self.messages.append(tool_results)
            final = message

        if final is None:
            return "(no response)"
        if final.stop_reason == "refusal":
            return "Sorry, I can't help with that request."
        return "".join(block.text for block in final.content if block.type == "text")


def main() -> None:
    assistant = RetailAssistant()
    print("RetailAssist is ready. Type 'quit' to exit.\n")
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
            reply = assistant.ask(user_text)
        except anthropic.AuthenticationError:
            print("Error: invalid or missing API key. Set ANTHROPIC_API_KEY in your environment.")
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
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
