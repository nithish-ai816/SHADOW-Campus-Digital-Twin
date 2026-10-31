"""
SHADOW - Camera Sources Architecture
Provides modular abstractions for video file input, USB webcams, RTSP/IP streams,
and simulated camera feeds with privacy and authorization guardrails.
"""

import os
import time
import math
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2

class BaseCameraSource(ABC):
    def __init__(self, camera_id: str, classroom_id: str, source_type: str):
        self.camera_id = camera_id
        self.classroom_id = classroom_id
        self.source_type = source_type
        self.is_authorized = False
        self.last_frame_time = 0.0
        self.frame_count = 0

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        pass

    @abstractmethod
    def release(self):
        pass

    def get_info(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "classroom_id": self.classroom_id,
            "source_type": self.source_type,
            "authorized": self.is_authorized,
            "frames_captured": self.frame_count,
            "last_frame_time": self.last_frame_time,
        }

class VideoFileSource(BaseCameraSource):
    """Reads uploaded or stored classroom video footage (MP4, AVI, MOV)."""
    def __init__(self, camera_id: str, classroom_id: str, file_path: str):
        super().__init__(camera_id, classroom_id, "video_file")
        self.file_path = file_path
        self.cap = None
        self.fps = 25.0
        self.total_frames = 0
        self.is_authorized = True

    def connect(self) -> bool:
        if not os.path.exists(self.file_path):
            return False
        self.cap = cv2.VideoCapture(self.file_path)
        if self.cap.isOpened():
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return True
        return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if not self.cap or not self.cap.isOpened():
            if not self.connect():
                return False, None, 0.0
        ret, frame = self.cap.read()
        if not ret:
            # Loop video for continuous monitoring
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return False, None, 0.0
        self.frame_count += 1
        self.last_frame_time = time.time()
        return True, frame, self.last_frame_time

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

class USBCameraSource(BaseCameraSource):
    """Connects to an authorized local USB camera / webcam."""
    def __init__(self, camera_id: str, classroom_id: str, device_index: int = 0, authorized: bool = False):
        super().__init__(camera_id, classroom_id, "usb_camera")
        self.device_index = device_index
        self.is_authorized = authorized
        self.cap = None

    def connect(self) -> bool:
        if not self.is_authorized:
            return False
        try:
            self.cap = cv2.VideoCapture(self.device_index)
            return self.cap.isOpened()
        except Exception:
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if not self.is_authorized or not self.cap or not self.cap.isOpened():
            return False, None, 0.0
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            self.last_frame_time = time.time()
        return ret, frame, self.last_frame_time

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

class RTSPCameraSource(BaseCameraSource):
    """Connects to authorized RTSP/IP camera or college CCTV streams."""
    def __init__(self, camera_id: str, classroom_id: str, rtsp_url: str, authorized: bool = False):
        super().__init__(camera_id, classroom_id, "rtsp_stream")
        self.rtsp_url = rtsp_url
        self.is_authorized = authorized
        self.cap = None

    def connect(self) -> bool:
        # STRICT SAFETY: College cameras require explicit authorized approval
        if not self.is_authorized:
            return False
        if not self.rtsp_url:
            return False
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            return self.cap.isOpened()
        except Exception:
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if not self.is_authorized or not self.cap or not self.cap.isOpened():
            return False, None, 0.0
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            self.last_frame_time = time.time()
        return ret, frame, self.last_frame_time

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

class SyntheticClassroomSource(BaseCameraSource):
    """
    High-fidelity synthetic classroom stream generator.
    Simulates various scenarios: Occupied class, Empty room with waste, Empty room clean.
    Ensures tests and prototypes run immediately without needing physical cameras.
    """
    def __init__(self, camera_id: str, classroom_id: str, scenario: str = "waste_scenario"):
        super().__init__(camera_id, classroom_id, "synthetic_feed")
        self.scenario = scenario
        self.is_authorized = True
        self.step = 0

    def connect(self) -> bool:
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        self.step += 1
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        img[:] = (22, 26, 36)

        # Classroom ceiling
        cv2.rectangle(img, (0, 0), (640, 75), (32, 40, 52), -1)

        # Scenario parameters
        if self.scenario == "waste_scenario":
            num_people = 0
            fans_on = 4
            lights_on = 6
        elif self.scenario == "occupied_class":
            num_people = 18
            fans_on = 4
            lights_on = 6
        elif self.scenario == "clean_standby":
            num_people = 0
            fans_on = 0
            lights_on = 0
        else:
            num_people = 2
            fans_on = 2
            lights_on = 2

        # Draw lights
        light_coords = [80, 180, 280, 380, 480, 580]
        for idx, lx in enumerate(light_coords):
            is_lit = idx < lights_on
            col = (210, 245, 255) if is_lit else (55, 65, 75)
            cv2.rectangle(img, (lx-25, 18), (lx+25, 32), col, -1)
            if is_lit:
                cv2.circle(img, (lx, 25), 10, (180, 230, 255), -1)

        # Draw fans
        fan_coords = [130, 230, 430, 530]
        angle = (self.step * 45) % 360
        for idx, fx in enumerate(fan_coords):
            is_spinning = idx < fans_on
            cv2.circle(img, (fx, 58), 7, (130, 145, 165), -1)
            rad = np.deg2rad(angle if is_spinning else 30)
            for arm in [0, 120, 240]:
                a = rad + np.deg2rad(arm)
                x2 = int(fx + 22 * math.cos(a))
                y2 = int(58 + 14 * math.sin(a))
                col = (190, 210, 230) if is_spinning else (90, 105, 120)
                cv2.line(img, (fx, 58), (x2, y2), col, 2)

        # Draw desks
        for row, dy in enumerate([150, 225, 300]):
            for col, dx in enumerate([110, 250, 390, 530]):
                cv2.rectangle(img, (dx-35, dy), (dx+35, dy+24), (55, 48, 42), -1)
                cv2.rectangle(img, (dx-25, dy+24), (dx+25, dy+40), (38, 32, 28), -1)

        # Draw people if present
        if num_people > 0:
            drawn = 0
            for row, dy in enumerate([140, 215, 290]):
                for col, dx in enumerate([110, 250, 390, 530]):
                    if drawn >= num_people:
                        break
                    # Person head & shoulders
                    cv2.circle(img, (dx, dy), 12, (180, 160, 140), -1)
                    cv2.ellipse(img, (dx, dy+22), (18, 14), 0, 0, 180, (70, 90, 140), -1)
                    drawn += 1

        # Feed banner
        cv2.putText(img, f"{self.camera_id} [{self.classroom_id}] - SIMULATED FEED",
                    (12, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (84, 214, 232), 1)

        self.frame_count += 1
        self.last_frame_time = time.time()
        return True, img, self.last_frame_time

    def release(self):
        pass
