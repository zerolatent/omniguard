from typing import List


INCIDENT_TYPES: List[str] = [
    "Fall Hazard (Height)",
    "Unguarded Floor/Wall Opening",
    "Unsecured Working Platform",
    "Equipment Instability/Tip-over Risk",
    "Heat/Cold Stress",
    "Person Down/Injured",
    "Medical Emergency - First Aid/Ambulance",
    "Electrical Hazard (Exposed Wire/Damage)",
    "Crush/Proximity Hazard (Blind Spot)",
    "PPE Violation (Missing/Incorrect)",
    "Laceration/Severe Bleeding",
    "Eye Injury/Foreign Object",
    "Muscle Strain/Sprain Injury",
    "Active Fire/Smoke",
    "Fire/Explosion Hazard",
    "Caught-in/Caught-between Machinery",
]


SAFETY_STATUS: List[str] = ["HIGH", "EXTREME"]


def derive_action_plan(incident_type: str, safety_status: str) -> str:
    incident = (incident_type or "").strip()
    status = (safety_status or "").strip().upper()

    if incident in (
        "Person Down/Injured",
        "Laceration/Severe Bleeding",
        "Medical Emergency - First Aid/Ambulance",
    ):
        return "CALL_EMS" if status == "EXTREME" else "NOTIFY_SUPERVISOR"

    if incident in ("Active Fire/Smoke", "Fire/Explosion Hazard"):
        return "EVACUATE"

    if incident in (
        "Electrical Hazard (Exposed Wire/Damage)",
        "Fall Hazard (Height)",
        "Unsecured Working Platform",
        "Unguarded Floor/Wall Opening",
        "Equipment Instability/Tip-over Risk",
        "Crush/Proximity Hazard (Blind Spot)",
    ):
        return "SHUTDOWN_EQUIPMENT" if status == "EXTREME" else "NOTIFY_SUPERVISOR"

    if incident in (
        "PPE Violation (Missing/Incorrect)",
        "Heat/Cold Stress",
        "Eye Injury/Foreign Object",
        "Muscle Strain/Sprain Injury",
        "Caught-in/Caught-between Machinery",
    ):
        return "NOTIFY_SUPERVISOR" if status == "HIGH" else "MONITOR"

    return "MONITOR"


