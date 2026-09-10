"""
main.py
-------
I use this as the entry point of my project. It starts a simple
command-line chat loop so I can ask the Multi-Tool Medical Agent
questions directly from the terminal.

Run:
    python main.py
"""

from src.agent import build_agent_executor


def main():
    print("=" * 60)
    print(" Multi-Tool Medical AI Agent")
    print(" Ask about Heart Disease / Cancer / Diabetes statistics,")
    print(" or ask general medical questions (definitions, symptoms, cures).")
    print(" Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    executor = build_agent_executor()

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            continue

        try:
            response = executor.invoke({"input": question})
            print(f"\nAgent: {response['output']}")
        except Exception as exc:  # resilience: never crash the chat loop
            print(f"\n[Error] Something went wrong while answering: {exc}")


if __name__ == "__main__":
    main()
