import json
import os
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

HARDWARE_DB_PATH = "data/hardware/components.json"

def load_db():
    with open(HARDWARE_DB_PATH, "r") as f:
        return json.load(f)

class HardwareQueryInput(BaseModel):
    use_case: str = Field(description="The use case: gaming, video_editing, machine_learning, office, 3d_rendering, music_production")
    device_type: str = Field(description="Either 'desktop' or 'laptop'")
    budget_tier: str = Field(description="Either 'low', 'mid', or 'high'")
    budget_max: float = Field(description="Maximum total budget in USD")

class HardwareTool(BaseTool):
    name: str = "hardware_recommender"
    description: str = (
        "Queries the hardware database and returns the best matching components "
        "for a desktop build or laptop based on use case, device type, budget tier, and max budget. "
        "Returns a JSON string with the recommended build and total price."
    )
    args_schema: type[BaseModel] = HardwareQueryInput

    def _run(self, use_case: str, device_type: str, budget_tier: str, budget_max: float) -> str:
        try:
            db = load_db()

            if device_type == "laptop":
                return self._recommend_laptop(db, use_case, budget_tier, budget_max)
            else:
                return self._recommend_desktop(db, use_case, budget_tier, budget_max)

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _recommend_laptop(self, db, use_case, budget_tier, budget_max):
        laptops = db["laptops"]

        candidates = [
            l for l in laptops
            if use_case in l["best_for"] and l["price"] <= budget_max
        ]

        if not candidates:
            candidates = [l for l in laptops if l["price"] <= budget_max]

        if not candidates:
            candidates = laptops

        tier_order = {"low": 0, "mid": 1, "high": 2}
        candidates.sort(key=lambda x: (
            abs(tier_order.get(x["tier"], 1) - tier_order.get(budget_tier, 1)),
            -x["price"]
        ))

        best = candidates[0]

        return json.dumps({
            "device_type": "laptop",
            "recommendation": best,
            "total_price": best["price"],
            "explanations": {
                "laptop": (
                    f"Selected {best['name']} because:\n"
                    f"  - {best['gpu']} delivers strong performance for {use_case}\n"
                    f"  - {best['display']} display suits the workload\n"
                    f"  - {best['ram']}GB RAM handles multitasking\n"
                    f"  - {best['battery']} battery for mobility\n"
                    f"  - Fits within the {budget_tier} budget tier at MAD {round(best['price']*10):,}"
                )
            }
        })

    def _recommend_desktop(self, db, use_case, budget_tier, budget_max):
        tier_order = {"low": 0, "mid": 1, "high": 2}

        def best_component(components):
            use_case_match = [
                c for c in components
                if use_case in c.get("best_for", [])
            ]
            pool = use_case_match if use_case_match else components
            pool.sort(key=lambda x: abs(
                tier_order.get(x.get("tier", "mid"), 1) -
                tier_order.get(budget_tier, 1)
            ))
            return pool[0]

        cpu = best_component(db["cpus"])
        gpu = best_component(db["gpus"])

        mb_candidates = [m for m in db["motherboards"] 
                        if m["socket"] == cpu["socket"] and m["ram_type"] == cpu["ram_type"]]
        if not mb_candidates:
            mb_candidates = [m for m in db["motherboards"] if m["socket"] == cpu["socket"]]
        if not mb_candidates:
            mb_candidates = db["motherboards"]
        mb = sorted(mb_candidates, key=lambda x: abs(
            tier_order.get(x["tier"], 1) - tier_order.get(budget_tier, 1)
        ))[0]

        ram_candidates = [r for r in db["ram"] if r["type"] == mb["ram_type"]]
        if not ram_candidates:
            ram_candidates = db["ram"]
        ram = sorted(ram_candidates, key=lambda x: abs(
            tier_order.get(x["tier"], 1) - tier_order.get(budget_tier, 1)
        ))[0]

        storage = best_component(db["storage"])
        psu = best_component(db["psus"])
        case = best_component(db["cases"])

        total = (
            cpu["price"] + gpu["price"] + mb["price"] +
            ram["price"] + storage["price"] + psu["price"] + case["price"]
        )

        build = {
            "cpu": cpu,
            "gpu": gpu,
            "motherboard": mb,
            "ram": ram,
            "storage": storage,
            "psu": psu,
            "case": case
        }

        explanations = {
            "cpu": (
                f"{cpu['name']} selected — {cpu['cores']} cores at "
                f"{cpu['boost_clock']}GHz boost, ideal for {use_case} workloads."
            ),
            "gpu": (
                f"{gpu['name']} selected — {gpu['vram']}GB VRAM provides "
                f"strong performance for {use_case}."
            ),
            "motherboard": (
                f"{mb['name']} selected — {mb['socket']} socket matches CPU, "
                f"supports {mb['ram_type']} up to {mb['max_ram']}GB."
            ),
            "ram": (
                f"{ram['name']} selected — {ram['capacity']}GB {ram['type']} "
                f"at {ram['speed']}MHz ensures smooth multitasking."
            ),
            "storage": (
                f"{storage['name']} selected — {storage['capacity']}GB "
                f"at {storage['read_speed']}MB/s read speed."
            ),
            "psu": (
                f"{psu['name']} selected — {psu['wattage']}W {psu['efficiency']} "
                f"covers system power draw with headroom."
            ),
            "case": (
                f"{case['name']} selected — {case['form_factor']} form factor "
                f"with {case['max_gpu_length']}mm GPU clearance."
            )
        }

        return json.dumps({
            "device_type": "desktop",
            "recommendation": build,
            "total_price": total,
            "within_budget": total <= budget_max,
            "explanations": explanations
        })