import json
from agents.orchestrator import run_pipeline

def main():
    print("="*60)
    print("Welcome to BuildMate - Multi-Agent PC Recommendation System")
    print("="*60)

    print("\nDevice type:")
    print("  1. Desktop")
    print("  2. Laptop")
    choice = input("Enter 1 or 2: ").strip()
    device_type = "desktop" if choice == "1" else "laptop"

    print("\nUse case:")
    print("  1. Gaming")
    print("  2. Video Editing")
    print("  3. Machine Learning")
    print("  4. Office")
    print("  5. 3D Rendering")
    print("  6. Music Production")
    use_case_map = {
        "1": "gaming", "2": "video_editing", "3": "machine_learning",
        "4": "office", "5": "3d_rendering", "6": "music_production"
    }
    use_choice = input("Enter 1-6: ").strip()
    use_case = use_case_map.get(use_choice, None)

    print("\nBudget tier:")
    print("  1. Low")
    print("  2. Mid")
    print("  3. High")
    tier_map = {"1": "low", "2": "mid", "3": "high"}
    tier_choice = input("Enter 1-3: ").strip()
    budget_tier = tier_map.get(tier_choice, None)

    budget_max = input("\nMaximum budget in USD (e.g. 1500): ").strip()
    try:
        budget_max = float(budget_max)
    except ValueError:
        print("Invalid budget, defaulting to 1500")
        budget_max = 1500.0

    free_text = input("\nDescribe your needs (or press Enter to skip): ").strip()
    if not free_text:
        free_text = f"I need a {device_type} for {use_case}"

    print("\n" + "="*60)
    print("Running BuildMate pipeline...")
    print("="*60 + "\n")

    result = run_pipeline(
        free_text=free_text,
        device_type_override=device_type,
        budget_max=budget_max,
        use_case_override=use_case,
        budget_tier_override=budget_tier
    )

    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()