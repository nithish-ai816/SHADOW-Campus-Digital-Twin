"""
SHADOW - External Automatic Shutdown Device Integration
Core architectural interface for communicating with physical electronic control devices
responsible for automatic circuit isolation.

SHADOW acts as the intelligent monitoring and decision-making brain.
The external device acts as the physical electrical actuator and relay controller.
Includes HMAC authentication, idempotent UUIDv4 command tracking, timeout handling,
fail-safe logic, circuit safety locks, emergency stop, and simulated hardware engine.
"""

import hmac
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple

# 10 Non-Negotiable Workflow States
class WorkflowState(str, Enum):
    MONITORING = "Monitoring"
    POTENTIAL_WASTE_DETECTED = "Potential energy waste detected"
    ALERT_GENERATED = "Alert generated"
    COUNTDOWN_ACTIVE = "Countdown active"
    CONDITION_CLEARED = "Condition cleared"
    SHUTDOWN_COMMAND_PENDING = "Shutdown command pending"
    COMMAND_SENT = "Command sent"
    DEVICE_ACK_RECEIVED = "Device acknowledgement received"
    SHUTDOWN_CONFIRMED = "Shutdown confirmed"
    COMMAND_FAILED = "Command failed"

class ProtocolType(str, Enum):
    HTTP_REST = "HTTP / REST API (JSON)"
    MQTT = "MQTT (Pub/Sub Broker)"
    MODBUS_TCP = "Modbus TCP (Industrial PLC)"
    WEBSOCKET = "Secure WebSocket (WSS)"
    SERIAL_UART = "Serial / UART COM (Microcontroller)"

class CircuitType:
    CONTROLLABLE = "Controllable Appliance Circuit"
    PROTECTED_SAFETY = "PROTECTED: Critical Safety / Emergency Circuit (LOCKED)"

DEFAULT_CIRCUITS = [
    {
        "circuit_id": "C1_FANS",
        "name": "Ceiling Fans Relay Bank",
        "type": CircuitType.CONTROLLABLE,
        "controllable": True,
        "status": "CLOSED (ENERGIZED)"
    },
    {
        "circuit_id": "C2_LIGHTS",
        "name": "Classroom Lighting Relay Bank",
        "type": CircuitType.CONTROLLABLE,
        "controllable": True,
        "status": "CLOSED (ENERGIZED)"
    },
    {
        "circuit_id": "C3_PROJECTOR",
        "name": "AV Projector Socket Relay",
        "type": CircuitType.CONTROLLABLE,
        "controllable": True,
        "status": "CLOSED (ENERGIZED)"
    },
    {
        "circuit_id": "C4_HVAC_AC",
        "name": "Air Conditioning Contactor",
        "type": CircuitType.CONTROLLABLE,
        "controllable": True,
        "status": "CLOSED (ENERGIZED)"
    },
    {
        "circuit_id": "C0_EMERGENCY_LIGHTS",
        "name": "Emergency Exit & Backup Lighting",
        "type": CircuitType.PROTECTED_SAFETY,
        "controllable": False,  # HARD LOCK: NEVER SHUTDOWN
        "status": "CLOSED (ENERGIZED)"
    },
    {
        "circuit_id": "C9_FIRE_ALARM",
        "name": "Life Safety / Smoke Detection Loop",
        "type": CircuitType.PROTECTED_SAFETY,
        "controllable": False,  # HARD LOCK: NEVER SHUTDOWN
        "status": "CLOSED (ENERGIZED)"
    }
]

DEFAULT_DEVICES = {
    "DEV-SHADOW-A101": {
        "device_id": "DEV-SHADOW-A101",
        "name": "Smart Power Controller Unit Alpha",
        "block": "A",
        "classroom_id": "A-101",
        "protocol": ProtocolType.HTTP_REST.value,
        "endpoint": "http://192.168.1.101:8080/api/v1/control",
        "auth_secret": "shadow_device_key_sec_a101_77a9",
        "connection_status": "Online",
        "latency_ms": 18,
        "firmware_version": "v2.4.1-sim",
        "circuits": [dict(c) for c in DEFAULT_CIRCUITS],
        "last_seen": "Just now",
        "last_command_id": None,
        "last_command_status": "Idle",
        "response_history": []
    },
    "DEV-SHADOW-A104": {
        "device_id": "DEV-SHADOW-A104",
        "name": "Smart Power Controller Unit Delta",
        "block": "A",
        "classroom_id": "A-104",
        "protocol": ProtocolType.HTTP_REST.value,
        "endpoint": "http://192.168.1.104:8080/api/v1/control",
        "auth_secret": "shadow_device_key_sec_a104_88b2",
        "connection_status": "Online",
        "latency_ms": 22,
        "firmware_version": "v2.4.1-sim",
        "circuits": [dict(c) for c in DEFAULT_CIRCUITS],
        "last_seen": "Just now",
        "last_command_id": None,
        "last_command_status": "Idle",
        "response_history": []
    },
    "DEV-SHADOW-B103": {
        "device_id": "DEV-SHADOW-B103",
        "name": "Smart Power Controller Unit Beta",
        "block": "B",
        "classroom_id": "B-103",
        "protocol": ProtocolType.MQTT.value,
        "endpoint": "mqtt://broker.campus.local:1883/devices/DEV-SHADOW-B103",
        "auth_secret": "shadow_device_key_sec_b103_33c1",
        "connection_status": "Online",
        "latency_ms": 14,
        "firmware_version": "v2.4.1-sim",
        "circuits": [dict(c) for c in DEFAULT_CIRCUITS],
        "last_seen": "Just now",
        "last_command_id": None,
        "last_command_status": "Idle",
        "response_history": []
    }
}

class ExternalDeviceManager:
    """
    Manages external automatic shutdown devices, authenticated dispatch,
    status receipt, and simulated device behavior.
    """
    def __init__(self, devices_dict: Optional[Dict[str, Any]] = None):
        self.devices: Dict[str, Dict[str, Any]] = devices_dict or {
            k: dict(v) for k, v in DEFAULT_DEVICES.items()
        }
        self.command_log: List[Dict[str, Any]] = []
        self.simulated_failure_mode: Optional[str] = None  # None | "timeout" | "auth_error" | "offline"

    def get_device_for_room(self, room_id: str) -> Optional[Dict[str, Any]]:
        for dev in self.devices.values():
            if dev.get("classroom_id") == room_id:
                return dev
        return None

    def sign_payload(self, payload: Dict[str, Any], secret: str) -> str:
        """Generates an HMAC-SHA256 signature for tamper-evident transmission."""
        canonical = json.dumps(payload, sort_keys=True)
        return hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()

    def build_command(
        self,
        device_id: str,
        action: str = "SHUTDOWN_CIRCUITS",
        target_circuit_ids: Optional[List[str]] = None,
        timeout_seconds: int = 15,
        reason: str = "Unoccupied room energy waste detected by Camera AI"
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Constructs an authenticated, idempotent, timeout-bound command.
        Applies safety filter: NEVER disconnects protected safety circuits.
        """
        device = self.devices.get(device_id)
        if not device:
            return None, f"Device {device_id} not registered."

        if target_circuit_ids is None:
            # Default to all controllable circuits
            target_circuit_ids = [
                c["circuit_id"] for c in device.get("circuits", []) if c.get("controllable", False)
            ]

        # STRICT SAFETY: Intercept and strip any protected life-safety circuits
        safe_circuits = []
        rejected_circuits = []
        for cid in target_circuit_ids:
            c_meta = next((c for c in device.get("circuits", []) if c["circuit_id"] == cid), None)
            if c_meta and c_meta.get("controllable", False):
                safe_circuits.append(cid)
            else:
                rejected_circuits.append(cid)

        if not safe_circuits and action == "SHUTDOWN_CIRCUITS":
            return None, "No controllable circuits selected. Safety circuits are locked from shutdown."

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=timeout_seconds)

        command_id = f"CMD-{uuid.uuid4().hex[:12].upper()}"

        payload = {
            "command_id": command_id,
            "device_id": device_id,
            "classroom_id": device.get("classroom_id"),
            "action": action,
            "target_circuits": safe_circuits,
            "protected_circuits_locked": [
                c["circuit_id"] for c in device.get("circuits", []) if not c.get("controllable", False)
            ],
            "reason": reason,
            "timestamp": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "issuer": "SHADOW_CAMPUS_AI_ENGINE",
            "is_simulation": True  # Non-negotiable label
        }

        # Sign with device auth secret
        signature = self.sign_payload(payload, device.get("auth_secret", "shadow_default"))
        command_package = {
            "payload": payload,
            "signature": signature,
            "auth_type": "HMAC-SHA256",
            "protocol": device.get("protocol")
        }

        return command_package, None

    def execute_command(
        self,
        command_package: Dict[str, Any],
        simulate_network: bool = True
    ) -> Dict[str, Any]:
        """
        Sends command to external device and waits for explicit acknowledgement.
        Never interprets a sent command as a confirmed shutdown without device confirmation.
        """
        t0 = time.time()
        payload = command_package["payload"]
        cmd_id = payload["command_id"]
        dev_id = payload["device_id"]
        action = payload["action"]
        device = self.devices.get(dev_id)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Check for simulated failure modes
        if self.simulated_failure_mode == "offline" or (device and device.get("connection_status") == "Disconnected"):
            resp = {
                "success": False,
                "workflow_state": WorkflowState.COMMAND_FAILED.value,
                "error": "DEVICE_UNREACHABLE: External device is disconnected or offline.",
                "command_id": cmd_id,
                "device_id": dev_id,
                "timestamp": now_str,
                "relay_status": "UNCHANGED",
                "ack_received": False,
                "confirmed_shutdown": False,
                "is_simulated": True
            }
            self._record_execution(cmd_id, dev_id, action, resp, latency_ms=0)
            return resp

        if self.simulated_failure_mode == "timeout":
            resp = {
                "success": False,
                "workflow_state": WorkflowState.COMMAND_FAILED.value,
                "error": "COMMUNICATION_TIMEOUT: External device failed to respond within 15 seconds. Failsafe activated.",
                "command_id": cmd_id,
                "device_id": dev_id,
                "timestamp": now_str,
                "relay_status": "UNKNOWN_FAILSAFE_HOLD",
                "ack_received": False,
                "confirmed_shutdown": False,
                "is_simulated": True
            }
            self._record_execution(cmd_id, dev_id, action, resp, latency_ms=15000)
            return resp

        if self.simulated_failure_mode == "auth_error":
            resp = {
                "success": False,
                "workflow_state": WorkflowState.COMMAND_FAILED.value,
                "error": "AUTHENTICATION_FAILED: Device rejected HMAC signature. Key mismatch.",
                "command_id": cmd_id,
                "device_id": dev_id,
                "timestamp": now_str,
                "relay_status": "REJECTED",
                "ack_received": True,
                "confirmed_shutdown": False,
                "is_simulated": True
            }
            self._record_execution(cmd_id, dev_id, action, resp, latency_ms=25)
            return resp

        # 2. Simulated Hardware Device Execution
        # If in normal simulated operation:
        target_circuits = payload.get("target_circuits", [])
        updated_circuit_statuses = {}

        if device:
            for c in device.get("circuits", []):
                cid = c["circuit_id"]
                if cid in target_circuits:
                    if action in ["SHUTDOWN_CIRCUITS", "EMERGENCY_STOP"]:
                        c["status"] = "OPEN (DISCONNECTED / DE-ENERGIZED)"
                    elif action == "RESTORE_CIRCUITS":
                        c["status"] = "CLOSED (ENERGIZED)"
                updated_circuit_statuses[cid] = c["status"]

            device["last_seen"] = now_str
            device["last_command_id"] = cmd_id
            device["last_command_status"] = "Success"

        exec_latency = 18.5  # ms simulated hardware relay switching time

        # Realistic Device Acknowledgement Receipt
        ack_receipt = {
            "device_id": dev_id,
            "command_id": cmd_id,
            "status": "EXECUTED",
            "executed_action": action,
            "circuits_switched": target_circuits,
            "current_circuit_states": updated_circuit_statuses,
            "relay_hardware": "SIMULATED_8CH_SSR_RELAY_BOARD",
            "microcontroller": "ESP32_WROOM_32D (SIMULATED)",
            "measured_current_amperes": 0.05 if action == "SHUTDOWN_CIRCUITS" else 4.2,
            "timestamp": now_str,
            "device_signature": hashlib.sha256(f"{cmd_id}:ACK_OK".encode()).hexdigest()[:16]
        }

        # ONLY confirm shutdown when device explicitly reports OPEN (DE-ENERGIZED)
        is_confirmed = False
        if action in ["SHUTDOWN_CIRCUITS", "EMERGENCY_STOP"]:
            is_confirmed = all(
                "OPEN" in updated_circuit_statuses.get(cid, "") for cid in target_circuits
            )

        resp = {
            "success": True,
            "workflow_state": WorkflowState.SHUTDOWN_CONFIRMED.value if is_confirmed else WorkflowState.DEVICE_ACK_RECEIVED.value,
            "message": f"External device successfully executed {action}. Circuits updated.",
            "command_id": cmd_id,
            "device_id": dev_id,
            "timestamp": now_str,
            "ack_receipt": ack_receipt,
            "ack_received": True,
            "confirmed_shutdown": is_confirmed,
            "latency_ms": exec_latency,
            "is_simulated": True,
            "safety_note": "SIMULATED OPERATION: Validated protocol communication only. Physical circuit isolation requires hardware."
        }

        self._record_execution(cmd_id, dev_id, action, resp, latency_ms=exec_latency)
        return resp

    def _record_execution(self, cmd_id: str, dev_id: str, action: str, resp: Dict[str, Any], latency_ms: float):
        record = {
            "command_id": cmd_id,
            "device_id": dev_id,
            "action": action,
            "success": resp["success"],
            "state": resp["workflow_state"],
            "timestamp": resp["timestamp"],
            "latency_ms": latency_ms,
            "error": resp.get("error"),
            "confirmed_shutdown": resp.get("confirmed_shutdown", False),
            "is_simulated": True
        }
        self.command_log.insert(0, record)
        if len(self.command_log) > 50:
            self.command_log.pop()

        dev = self.devices.get(dev_id)
        if dev:
            hist = dev.get("response_history", [])
            hist.insert(0, record)
            if len(hist) > 30:
                hist.pop()
            dev["response_history"] = hist

    def manual_emergency_stop(self, room_id: Optional[str] = None) -> Dict[str, Any]:
        """
        EMERGENCY STOP: Immediately shuts down all controllable circuits across the
        target classroom or the entire campus. Safety circuits remain protected.
        """
        results = []
        target_devs = [self.get_device_for_room(room_id)] if room_id else list(self.devices.values())
        for dev in target_devs:
            if not dev:
                continue
            pkg, err = self.build_command(
                dev["device_id"],
                action="EMERGENCY_STOP",
                reason="OPERATOR EMERGENCY STOP ENGAGED"
            )
            if pkg:
                res = self.execute_command(pkg)
                results.append(res)
        return {"action": "EMERGENCY_STOP", "results": results, "timestamp": datetime.now().isoformat()}

    def manual_reset_restore(self, room_id: Optional[str] = None) -> Dict[str, Any]:
        """Manual override: Restores circuits back to energized (normal operation)."""
        results = []
        target_devs = [self.get_device_for_room(room_id)] if room_id else list(self.devices.values())
        for dev in target_devs:
            if not dev:
                continue
            pkg, err = self.build_command(
                dev["device_id"],
                action="RESTORE_CIRCUITS",
                reason="OPERATOR MANUAL RESTORE OVERRIDE"
            )
            if pkg:
                res = self.execute_command(pkg)
                results.append(res)
        return {"action": "RESTORE_CIRCUITS", "results": results, "timestamp": datetime.now().isoformat()}

# Global singleton device manager
device_manager = ExternalDeviceManager()
