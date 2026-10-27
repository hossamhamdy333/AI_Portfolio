def cli_approval(draft, verdict):
    print("\n--- Answer ---")
    print(draft)
    print("\n--- Critic ---")
    print("approved:", verdict.approved)
    print("feedback:", verdict.feedback)

    while True:
        choice = input("\nApprove this answer? [y/n]: ").strip().lower()
        if choice in ("y", "n"):
            break
        print("Type y or n.")

    note = input("Note (optional, press Enter to skip): ").strip()
    return choice == "y", note
