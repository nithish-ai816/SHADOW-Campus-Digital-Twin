"""
SHADOW - Automatic Energy-Waste Estimation Engine
Combines camera-based AI detections, classroom schedules, visible appliance states,
and configured electrical ratings to identify unnecessary electricity waste.

IMPORTANT: Strictly estimation-based. Distinguishes estimated conditions from
confirmed electrical measurements.
"""

from datetime import datetime, time as dtime
from typing import Dict, Any, List, Tuple, Optional

# Default appliance power ratings in Watts
DEFAULT_POWER_RATINGS = {
    "ceiling_fan_w": 75.0,        # 75W standard BLDC / Induction ceiling fan
    "tube_light_w": 40.0,         # 40W standard fluorescent/LED fixture
    "projector_w": 250.0,         # 250W classroom projector
    "split_ac_w": 1500.0,         # 1500W (1.5 Ton) split air conditioner
    "computer_w": 150.0,          # 150W desktop workstation
}

# Standard campus classroom timetable slots
DEFAULT_SCHEDULES = [
    {"slot": "Period 1 (Morning Lecture)",  "start": "09:00", "end": "10:00", "type": "Scheduled Class"},
    {"slot": "Period 2 (Morning Lecture)",  "start": "10:00", "end": "11:00", "type": "Scheduled Class"},
    {"slot": "Tea / Recess Break",          "start": "11:00", "end": "11:30", "type": "Unscheduled / Break"},
    {"slot": "Period 3 (Laboratory/Hands-on)","start": "11:30", "end": "13:00", "type": "Scheduled Class"},
    {"slot": "Lunch Interval",              "start": "13:00", "end": "14:00", "type": "Unscheduled / Break"},
    {"slot": "Period 4 (Afternoon Lecture)","start": "14:00", "end": "15:30", "type": "Scheduled Class"},
    {"slot": "Period 5 (Tutorial / Seminar)", "start": "15:30", "end": "16:30", "type": "Scheduled Class"},
    {"slot": "Evening / After-Hours",       "start": "16:30", "end": "22:00", "type": "Unscheduled / Free"},
]

class EnergyWasteEngine:
    def __init__(self, power_ratings: Optional[Dict[str, float]] = None):
        self.ratings = power_ratings or DEFAULT_POWER_RATINGS.copy()
        self.tariff_per_kwh = 8.00  # Currency units per kWh
        self.co2_factor_kg_per_kwh = 0.82  # Central grid emissions factor

        # Duration tracker: {room_id: {"start_time": datetime, "accumulated_minutes": float}}
        self.waste_duration_tracker: Dict[str, Dict[str, Any]] = {}

    def get_schedule_status(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Determines current scheduled slot and whether the room is scheduled."""
        if now is None:
            now = datetime.now()
        current_time_str = now.strftime("%H:%M")

        for item in DEFAULT_SCHEDULES:
            if item["start"] <= current_time_str < item["end"]:
                is_scheduled = (item["type"] == "Scheduled Class")
                return {
                    "current_slot": item["slot"],
                    "start": item["start"],
                    "end": item["end"],
                    "type": item["type"],
                    "is_scheduled": is_scheduled
                }

        # Fallback (night / off-peak)
        return {
            "current_slot": "Night / Closed Hours",
            "start": "22:00",
            "end": "08:30",
            "type": "Unscheduled / Free",
            "is_scheduled": False
        }

    def evaluate(
        self,
        ai_data: Dict[str, Any],
        room_config: Optional[Dict[str, Any]] = None,
        waste_threshold_kw: float = 0.15,
        waste_minutes_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Combines AI vision findings, schedule status, and configured appliance ratings
        to compute estimated room-wise and appliance-wise energy waste.
        """
        room_id = ai_data.get("classroom_id", "A-101")
        people_count = ai_data.get("people_count", 0)
        occupancy_status = ai_data.get("occupancy_status", "Unoccupied")
        now_dt = datetime.now()

        schedule_info = self.get_schedule_status(now_dt)
        is_scheduled = schedule_info["is_scheduled"]

        fans_info = ai_data.get("fans", {"on": 0, "total": 4, "confidence": 0.90})
        lights_info = ai_data.get("lights", {"on": 0, "total": 6, "confidence": 0.95})
        proj_info = ai_data.get("projector", {"on": 0, "total": 1, "confidence": 0.88})
        ac_info = ai_data.get("ac", {"on": 0, "total": 1, "confidence": 0.86})

        fans_on = fans_info["on"]
        lights_on = lights_info["on"]
        proj_on = proj_info["on"]
        ac_on = ac_info["on"]

        # Compute power loads per appliance
        fan_kw = round((fans_on * self.ratings.get("ceiling_fan_w", 75.0)) / 1000.0, 3)
        light_kw = round((lights_on * self.ratings.get("tube_light_w", 40.0)) / 1000.0, 3)
        proj_kw = round((proj_on * self.ratings.get("projector_w", 250.0)) / 1000.0, 3)
        ac_kw = round((ac_on * self.ratings.get("split_ac_w", 1500.0)) / 1000.0, 3)

        total_active_load_kw = round(fan_kw + light_kw + proj_kw + ac_kw, 3)

        # Waste Classification Rules
        is_waste = False
        waste_power_kw = 0.0
        reason = "Normal operation: room occupancy aligns with energy usage."
        severity = "Normal"
        confidence_factors = [
            ai_data.get("occupancy_confidence", 0.92),
            fans_info.get("confidence", 0.90),
            lights_info.get("confidence", 0.94),
        ]
        overall_confidence = round(sum(confidence_factors) / len(confidence_factors), 2)

        # CONDITION 1: Empty Room during Unscheduled Period (High Waste)
        if people_count == 0 and not is_scheduled and total_active_load_kw > waste_threshold_kw:
            is_waste = True
            waste_power_kw = total_active_load_kw
            severity = "High"
            reason = (
                f"Classroom {room_id} is completely unoccupied (0 occupants) during unscheduled period "
                f"({schedule_info['current_slot']}), but {fans_on} fans, {lights_on} lights"
                f"{', projector' if proj_on else ''}{', AC' if ac_on else ''} remain running."
            )

        # CONDITION 2: Empty Room during Scheduled Class (Warning Waste - Early dismissal/cancelled)
        elif people_count == 0 and is_scheduled and total_active_load_kw > waste_threshold_kw:
            is_waste = True
            waste_power_kw = total_active_load_kw
            severity = "Warning"
            reason = (
                f"Classroom {room_id} has scheduled class ({schedule_info['current_slot']}) but 0 occupants "
                f"detected by camera AI. {fans_on} fans and {lights_on} lights running without attendance."
            )

        # CONDITION 3: Low occupancy with excessive appliances running
        elif 0 < people_count <= 2 and total_active_load_kw > 1.0 and (fans_on >= 4 or lights_on >= 6):
            # Partial unnecessary consumption
            surplus_fans = max(0, fans_on - 1)
            surplus_lights = max(0, lights_on - 2)
            surplus_kw = round(
                (surplus_fans * self.ratings.get("ceiling_fan_w", 75.0) +
                 surplus_lights * self.ratings.get("tube_light_w", 40.0)) / 1000.0, 3
            )
            if surplus_kw > waste_threshold_kw:
                is_waste = True
                waste_power_kw = surplus_kw
                severity = "Warning"
                reason = (
                    f"Only {people_count} occupant(s) in {room_id}, but full capacity lighting and fans "
                    f"are engaged ({fans_on} fans, {lights_on} lights). Surplus load: {surplus_kw} kW."
                )

        # Track duration
        if is_waste:
            if room_id not in self.waste_duration_tracker:
                self.waste_duration_tracker[room_id] = {
                    "start_time": now_dt,
                    "accumulated_minutes": 25.0  # Initial realistic estimate for prototype
                }
            tracker = self.waste_duration_tracker[room_id]
            elapsed_min = (now_dt - tracker["start_time"]).total_seconds() / 60.0
            total_waste_minutes = round(tracker["accumulated_minutes"] + elapsed_min, 1)
        else:
            if room_id in self.waste_duration_tracker:
                del self.waste_duration_tracker[room_id]
            total_waste_minutes = 0.0

        if waste_minutes_override is not None:
            total_waste_minutes = waste_minutes_override

        # Energy waste in kWh = kW * (minutes / 60)
        hours_wasted = total_waste_minutes / 60.0
        wasted_kwh = round(waste_power_kw * hours_wasted, 3)
        cost_waste = round(wasted_kwh * self.tariff_per_kwh, 2)
        co2_waste_kg = round(wasted_kwh * self.co2_factor_kg_per_kwh, 2)

        # Appliance-wise breakdown
        appliance_breakdown = {
            "fans": {
                "count_on": fans_on,
                "power_kw": fan_kw,
                "waste_kw": fan_kw if is_waste and people_count == 0 else 0.0,
                "rating_w": self.ratings.get("ceiling_fan_w", 75.0)
            },
            "lights": {
                "count_on": lights_on,
                "power_kw": light_kw,
                "waste_kw": light_kw if is_waste and people_count == 0 else 0.0,
                "rating_w": self.ratings.get("tube_light_w", 40.0)
            },
            "projector": {
                "count_on": proj_on,
                "power_kw": proj_kw,
                "waste_kw": proj_kw if is_waste and people_count == 0 else 0.0,
                "rating_w": self.ratings.get("projector_w", 250.0)
            },
            "ac": {
                "count_on": ac_on,
                "power_kw": ac_kw,
                "waste_kw": ac_kw if is_waste and people_count == 0 else 0.0,
                "rating_w": self.ratings.get("split_ac_w", 1500.0)
            }
        }

        return {
            "classroom_id": room_id,
            "timestamp": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "is_waste": is_waste,
            "severity": severity,
            "waste_power_kw": waste_power_kw,
            "total_active_load_kw": total_active_load_kw,
            "wasted_minutes": total_waste_minutes,
            "estimated_wasted_kwh": wasted_kwh,
            "estimated_cost_waste": cost_waste,
            "estimated_co2_kg": co2_waste_kg,
            "reason": reason,
            "confidence": overall_confidence,
            "occupancy_status": occupancy_status,
            "people_count": people_count,
            "schedule": schedule_info,
            "appliance_breakdown": appliance_breakdown,
            "is_estimated": True,
            "measurement_note": "ESTIMATED VALUE: Derived from AI camera vision and timetable rules. Not a physical meter reading."
        }

# Global singleton waste engine
waste_engine = EnergyWasteEngine()
