"""
main.py
-------
I use this as the entry point of my project. It starts a simple
command-line chat loop so I can ask the Multi-Tool Medical Agent
questions directly from the terminal.

Run:
    python main.py
"""

import traceback

from src.agent import TOOLS, build_agent_executor


def main():
    print("=" * 60)
    print(" Multi-Tool Medical AI Agent")
    print(" Ask about Heart Disease / Cancer / Diabetes statistics,")
    print(" or ask general medical questions (definitions, symptoms, cures).")
    print(" Type 'exit' or 'quit' to stop.")
    print("=" * 60)
    print(" Tools loaded:", ", ".join(t.name for t in TOOLS))

    agent = build_agent_executor()

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            continue

        try:
            result = agent.invoke({"messages": [{"role": "user", "content": question}]})
            content = result["messages"][-1].content
            if isinstance(content, list):
                text = "".join(
                    block.get("text", "") for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                text = content
            print(f"\nAgent: {text}")
        except Exception:
            print("\n[Error] Something went wrong while answering. Full details below:")
            traceback.print_exc()


if __name__ == "__main__":
    main()