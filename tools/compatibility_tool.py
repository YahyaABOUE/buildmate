import json
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class CompatibilityInput(BaseModel):
    build_json: str = Field(description="JSON string of the recommended desktop build to validate")
    budget_max: float = Field(description="Maximum budget in USD to validate against total price")

class CompatibilityTool(BaseTool):
    name: str = "compatibility_checker"
    description: str = (
        "Validates a desktop PC build for hardware compatibility issues. "
        "Checks CPU socket, RAM type, PSU wattage, and budget. "
        "Returns a JSON string with passed checks, failed checks, and whether the build is valid."
    )
    args_schema: type[BaseModel] = CompatibilityInput

    def _run(self, build_json: str, budget_max: float) -> str:
        try:
            data = json.loads(build_json)

            if "recommendation" in data:
                build = data["recommendation"]
                total_price = data.get("total_price", 0)
            else:
                build = data
                total_price = sum(
                    c.get("price", 0) for c in build.values()
                    if isinstance(c, dict)
                )

            if data.get("device_type") == "laptop":
                return json.dumps({
                    "valid": True,
                    "passed": ["Laptop builds do not require compatibility checks"],
                    "failed": [],
                    "warnings": []
                })

            passed = []
            failed = []
            warnings = []

            cpu = build.get("cpu", {})
            mb = build.get("motherboard", {})
            ram = build.get("ram", {})
            gpu = build.get("gpu", {})
            psu = build.get("psu", {})
            case = build.get("case", {})

            # Check 1: CPU socket vs motherboard socket
            if cpu.get("socket") and mb.get("socket"):
                if cpu["socket"] == mb["socket"]:
                    passed.append(f"CPU socket match: {cpu['socket']} compatible with {mb['name']}")
                else:
                    failed.append(
                        f"CPU socket mismatch: CPU uses {cpu['socket']} but motherboard uses {mb['socket']}"
                    )

            # Check 2: RAM type vs motherboard RAM type
            if ram.get("type") and mb.get("ram_type"):
                if ram["type"] == mb["ram_type"]:
                    passed.append(f"RAM type match: {ram['type']} compatible with {mb['name']}")
                else:
                    failed.append(
                        f"RAM type mismatch: RAM is {ram['type']} but motherboard supports {mb['ram_type']}"
                    )

            # Check 3: CPU ram_type vs motherboard RAM type
            if cpu.get("ram_type") and mb.get("ram_type"):
                if cpu["ram_type"] == mb["ram_type"]:
                    passed.append(f"CPU RAM type match: CPU platform supports {cpu['ram_type']}")
                else:
                    failed.append(
                        f"CPU RAM type mismatch: CPU platform uses {cpu['ram_type']} but motherboard uses {mb['ram_type']}"
                    )

            # Check 4: PSU wattage vs total component TDP
            cpu_tdp = cpu.get("tdp", 0)
            gpu_tdp = gpu.get("tdp", 0)
            total_tdp = cpu_tdp + gpu_tdp + 100
            psu_wattage = psu.get("wattage", 0)

            if psu_wattage >= total_tdp:
                passed.append(
                    f"PSU wattage sufficient: {psu_wattage}W covers estimated {total_tdp}W draw"
                )
            else:
                failed.append(
                    f"PSU wattage insufficient: {psu_wattage}W is too low for estimated {total_tdp}W draw"
                )

            # Check 5: RAM capacity vs motherboard max RAM
            ram_capacity = ram.get("capacity", 0)
            mb_max_ram = mb.get("max_ram", 128)
            if ram_capacity <= mb_max_ram:
                passed.append(
                    f"RAM capacity OK: {ram_capacity}GB within motherboard max of {mb_max_ram}GB"
                )
            else:
                failed.append(
                    f"RAM capacity exceeded: {ram_capacity}GB exceeds motherboard max of {mb_max_ram}GB"
                )

            # Check 6: Budget validation
            if total_price <= budget_max:
                passed.append(
                    f"Budget OK: Total MAD {round(total_price*10):,} is within MAD {round(budget_max*10):,} budget"
                )
            else:
                failed.append(
                    f"Budget exceeded: Total MAD {round(total_price*10):,} exceeds MAD {round(budget_max*10):,} budget"
                )

            # Warning: GPU TDP is very high
            if gpu_tdp >= 400:
                warnings.append(
                    f"High GPU TDP ({gpu_tdp}W): Ensure case has adequate airflow"
                )

            # Warning: No NVMe storage
            storage = build.get("storage", {})
            if storage.get("type") == "HDD":
                warnings.append("HDD detected: Consider upgrading to NVMe SSD for better performance")

            is_valid = len(failed) == 0

            return json.dumps({
                "valid": is_valid,
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "total_price": total_price,
                "estimated_tdp": total_tdp
            })

        except Exception as e:
            return json.dumps({"error": str(e), "valid": False})