from agent import chat, extract_memory
from memory import save_memory

# Entry point for the chat agent.
# Handles user interaction, response generation,
# memory extraction, and memory storage.

print("Agent ready")

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    response = chat(user_input)

    print("\nAssistant:", response)

    memory = extract_memory(user_input)

    if (memory and memory.get("remember")):
        save_memory(memory["memory"])
        print(f"\n[Memory Saved] {memory['memory']}")