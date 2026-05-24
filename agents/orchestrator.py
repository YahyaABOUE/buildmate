import json
import os
from datetime import datetime
from tools.classifier_tool import ClassifierTool
from tools.hardware_tool import HardwareTool
from tools.compatibility_tool import CompatibilityTool

import re

def is_english(text: str) -> bool:
    # Check for Arabic characters
    if re.search(r'[\u0600-\u06FF]', text):
        return False
    # Check for common French words
    french_words = ['je', 'tu', 'il', 'nous', 'vous', 'ils', 'mon', 'ton', 'son',
                    'pour', 'avec', 'une', 'des', 'les', 'est', 'sont', 'avoir',
                    'besoin', 'veux', 'voudrais', 'cherche', 'ordinateur']
    words = text.lower().split()
    french_count = sum(1 for w in words if w in french_words)
    if french_count >= 2:
        return False
    return True

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.75

def get_logger():
    log_path = os.path.join(LOG_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    actions = []

    def log(agent, action, details):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "details": details
        }
        actions.append(entry)
        with open(log_path, "w") as f:
            json.dump(actions, f, indent=2)
        print(f"[LOG] {entry['timestamp']} | {agent} | {action}")

    return log

def human_in_the_loop(recommendation: dict, low_confidence_fields: list) -> dict:
    print("\n" + "="*60)
    print("HUMAN APPROVAL CHECKPOINT")
    print("="*60)

    if low_confidence_fields:
        print(f"\n⚠️  Low confidence detected on: {', '.join(low_confidence_fields)}")
        print("Please review the recommendation carefully.\n")

    if recommendation.get("device_type") == "laptop":
        laptop = recommendation.get("recommendation", {})
        print(f"\nRecommended Laptop: {laptop.get('name')}")
        print(f"CPU:      {laptop.get('cpu')}")
        print(f"GPU:      {laptop.get('gpu')}")
        print(f"RAM:      {laptop.get('ram')}GB")
        print(f"Storage:  {laptop.get('storage')}GB")
        print(f"Display:  {laptop.get('display')}")
        print(f"Battery:  {laptop.get('battery')}")
        print(f"Weight:   {laptop.get('weight')}kg")
        print(f"Price:    ${recommendation.get('total_price')}")
    else:
        build = recommendation.get("recommendation", {})
        print(f"\nRecommended Desktop Build:")
        for component, details in build.items():
            if isinstance(details, dict):
                print(f"  {component.upper():<15} {details.get('name')} — ${details.get('price')}")
        print(f"\nTotal Price:    ${recommendation.get('total_price')}")
        print(f"Within Budget:  {recommendation.get('within_budget')}")

    print("\nExplanations:")
    for component, explanation in recommendation.get("explanations", {}).items():
        print(f"  • {explanation}")

    print("\n" + "="*60)
    print("Options:")
    print("  1. Approve recommendation")
    print("  2. Change budget")
    print("  3. Switch device type")
    print("  4. Change use case")
    print("  5. Reject")
    print("="*60)

    while True:
        choice = input("Enter your choice (1-5): ").strip()
        if choice == "1":
            return {"action": "approve"}
        elif choice == "2":
            new_budget = input("Enter new maximum budget in USD: ").strip()
            try:
                return {"action": "change_budget", "new_budget": float(new_budget)}
            except ValueError:
                print("Invalid budget, try again.")
        elif choice == "3":
            new_device = input("Enter device type (desktop/laptop): ").strip().lower()
            if new_device in ["desktop", "laptop"]:
                return {"action": "change_device", "new_device": new_device}
            else:
                print("Invalid choice.")
        elif choice == "4":
            print("Use cases: gaming, video_editing, machine_learning, office, 3d_rendering, music_production")
            new_use_case = input("Enter use case: ").strip().lower()
            return {"action": "change_use_case", "new_use_case": new_use_case}
        elif choice == "5":
            return {"action": "reject"}
        else:
            print("Please enter 1-5.")

def run_pipeline(
    free_text: str,
    device_type_override: str,
    budget_max: float,
    use_case_override: str = None,
    budget_tier_override: str = None,
    preferences: list = None,
    skip_hitl: bool = False
):
    log = get_logger()
    log("orchestrator", "pipeline_started", {
        "free_text": free_text,
        "device_type_override": device_type_override,
        "budget_max": budget_max,
        "use_case_override": use_case_override,
        "budget_tier_override": budget_tier_override,
        "preferences": preferences or []
    })

    # Step 1: NLP Classification
    classification_result = None
    low_confidence_fields = []
    if free_text.strip():
        if not is_english(free_text):
            return {
                "success": False,
                "error": "Please describe your needs in English. Arabic and French are not supported yet."
            }
        log("nlp_agent", "classification_started", {"input": free_text})
        try:
            classifier_tool = ClassifierTool()
            raw = classifier_tool._run(text=free_text)
            classification_result = json.loads(raw)
            log("nlp_agent", "classification_complete", classification_result)

            for field, result in classification_result.items():
                if result.get("confidence", 1.0) < CONFIDENCE_THRESHOLD:
                    low_confidence_fields.append(field)
                    log("orchestrator", "low_confidence_detected", {
                        "field": field,
                        "confidence": result.get("confidence")
                    })
            if classification_result:
                avg_confidence = sum(
                    classification_result[f].get("confidence", 0)
                    for f in classification_result
                ) / len(classification_result)
                use_case_conf = classification_result.get("use_case", {}).get("confidence", 0)
                if avg_confidence < 0.55 or use_case_conf < 0.4:
                    log("orchestrator", "input_rejected", {"reason": f"Confidence too low: avg={avg_confidence:.2f} use_case={use_case_conf:.2f}"})
                    return {
                        "success": False,
                        "error": "Your input was too ambiguous to process. Please describe your needs more clearly."
                    }

        except Exception as e:
            log("nlp_agent", "classification_failed", {"error": str(e)})
            classification_result = None

    # Step 2: Resolve labels — explicit overrides ALWAYS win over model
    if use_case_override:
        use_case = use_case_override
    elif classification_result:
        use_case = classification_result["use_case"]["label"]
    else:
        use_case = "office"

    # Device type: explicit UI selection ALWAYS wins over model prediction
    device_type = device_type_override if device_type_override else (
        classification_result["device_type"]["label"] if classification_result else "desktop"
    )

    if budget_tier_override:
        budget_tier = budget_tier_override
    elif classification_result:
        budget_tier = classification_result["budget_tier"]["label"]
    else:
        budget_tier = "mid"

    log("orchestrator", "labels_resolved", {
        "use_case": use_case,
        "device_type": device_type,
        "budget_tier": budget_tier,
        "budget_max": budget_max,
        "overrides_applied": {
            "use_case": use_case_override is not None,
            "device_type": device_type_override is not None,
            "budget_tier": budget_tier_override is not None
        }
    })

    max_retries = 3

    # Step 3: Recommendation + compatibility + HITL loop
    for attempt in range(max_retries):
        log("recommendation_agent", f"recommendation_attempt_{attempt+1}", {
            "use_case": use_case,
            "device_type": device_type,
            "budget_tier": budget_tier,
            "budget_max": budget_max
        })

        try:
            hardware_tool = HardwareTool()
            raw = hardware_tool._run(
                use_case=use_case,
                device_type=device_type,
                budget_tier=budget_tier,
                budget_max=budget_max
            )
            recommendation = json.loads(raw)
            log("recommendation_agent", "recommendation_complete", {
                "total_price": recommendation.get("total_price")
            })
        except Exception as e:
            log("recommendation_agent", "recommendation_failed", {"error": str(e)})
            return {"error": "Recommendation failed", "success": False}

        # Step 4: Compatibility check
        if device_type == "desktop":
            try:
                compatibility_tool = CompatibilityTool()
                raw = compatibility_tool._run(
                    build_json=json.dumps(recommendation),
                    budget_max=budget_max
                )
                compatibility_result = json.loads(raw)
                print(f"[COMPAT] valid={compatibility_result.get('valid')} failed={compatibility_result.get('failed',[])}")
                print(f"[COMPAT] total_price={compatibility_result.get('total_price')} budget_max={budget_max}")
                log("compatibility_agent", "compatibility_complete", {
                    "valid": compatibility_result.get("valid"),
                    "failed": compatibility_result.get("failed", [])
                })

                if not compatibility_result.get("valid"):
                    failed_checks = compatibility_result.get("failed", [])
                    log("orchestrator", "compatibility_failed_retrying", {
                        "issues": failed_checks
                    })

                    budget_exceeded = any("budget" in f.lower() for f in failed_checks)
                    socket_mismatch = any("socket" in f.lower() for f in failed_checks)
                    ram_mismatch = any("ram" in f.lower() for f in failed_checks)
                    psu_issue = any(
                        "psu" in f.lower() or "wattage" in f.lower()
                        for f in failed_checks
                    )

                    if budget_exceeded:
                        actual_total = compatibility_result.get("total_price", budget_max)
                        budget_max = max(budget_max * 1.5, actual_total * 1.1)
                        log("orchestrator", "budget_relaxed", {
                            "new_budget_max": budget_max
                        })
                    elif socket_mismatch or ram_mismatch:
                        if budget_tier == "low":
                            budget_tier = "mid"
                        elif budget_tier == "mid":
                            budget_tier = "high"
                        log("orchestrator", "tier_escalated", {
                            "new_tier": budget_tier
                        })
                    elif psu_issue:
                        if budget_tier == "low":
                            budget_tier = "mid"
                        log("orchestrator", "tier_escalated_for_psu", {
                            "new_tier": budget_tier
                        })
                    else:
                        if budget_tier == "low":
                            budget_tier = "mid"
                        elif budget_tier == "mid":
                            budget_tier = "high"

                    continue

            except Exception as e:
                log("compatibility_agent", "compatibility_error", {"error": str(e)})
                compatibility_result = {"valid": False, "error": str(e)}
        else:
            compatibility_result = {
                "status": "not_applicable",
                "reason": "Laptop recommendations are preconfigured systems and do not require component-level compatibility checks."
            }

        # Step 5: HITL checkpoint
        log("orchestrator", "hitl_checkpoint_reached", {
            "low_confidence_fields": low_confidence_fields
        })

        if skip_hitl:
            log("orchestrator", "hitl_skipped", {"reason": "UI handles approval"})
            all_overridden = use_case_override and device_type_override and budget_tier_override
            return {
                "success": True,
                "approved": True,
                "classification": classification_result,
                "labels": {
                    "use_case": use_case,
                    "device_type": device_type,
                    "budget_tier": budget_tier
                },
                "recommendation": recommendation,
                "compatibility": compatibility_result,
                "low_confidence_fields": low_confidence_fields
            }

        hitl_response = human_in_the_loop(recommendation, low_confidence_fields)

        if hitl_response["action"] == "approve":
            log("orchestrator", "recommendation_approved", {})
            return {
                "success": True,
                "approved": True,
                "classification": classification_result,
                "labels": {
                    "use_case": use_case,
                    "device_type": device_type,
                    "budget_tier": budget_tier
                },
                "recommendation": recommendation,
                "compatibility": compatibility_result,
                "low_confidence_fields": low_confidence_fields
            }

        elif hitl_response["action"] == "change_budget":
            budget_max = hitl_response["new_budget"]
            log("orchestrator", "budget_updated", {"new_budget": budget_max})

        elif hitl_response["action"] == "change_device":
            device_type = hitl_response["new_device"]
            log("orchestrator", "device_type_updated", {"new_device": device_type})

        elif hitl_response["action"] == "change_use_case":
            use_case = hitl_response["new_use_case"]
            log("orchestrator", "use_case_updated", {"new_use_case": use_case})

        elif hitl_response["action"] == "reject":
            log("orchestrator", "recommendation_rejected", {})
            return {
                "success": True,
                "approved": False,
                "message": "Recommendation rejected by user."
            }

    return {"error": "Max retries reached", "success": False}