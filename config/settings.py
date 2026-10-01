import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load local .env file if it exists
load_dotenv()

class DroidCamSettings(BaseModel):
    enabled: bool = Field(default_factory=lambda: os.getenv("DROIDCAM_ENABLED", "True").lower() == "true")
    host: str = Field(default_factory=lambda: os.getenv("DROIDCAM_HOST", "10.140.159.218"))
    port: int = Field(default_factory=lambda: int(os.getenv("DROIDCAM_PORT", "4747")))
    video_path: str = Field(default_factory=lambda: os.getenv("DROIDCAM_VIDEO_PATH", "/video"))

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def video_url(self) -> str:
        vpath = self.video_path if self.video_path.startswith("/") else f"/{self.video_path}"
        return f"http://{self.host}:{self.port}{vpath}"

class ESP32CamSettings(BaseModel):
    enabled: bool = Field(default_factory=lambda: os.getenv("ESP32_ENABLED", "True").lower() == "true")
    stream_url: str = Field(default_factory=lambda: os.getenv("ESP32_STREAM_URL", "http://192.168.1.100:81/stream"))

class CameraRetrySettings(BaseModel):
    enabled: bool = Field(default_factory=lambda: os.getenv("CAMERA_RETRY_ENABLED", "True").lower() == "true")
    max_attempts: int = Field(default_factory=lambda: int(os.getenv("CAMERA_RETRY_MAX", "5")))
    delay_seconds: float = Field(default_factory=lambda: float(os.getenv("CAMERA_RETRY_DELAY", "2.0")))

class PCCameraSettings(BaseModel):
    device_index: int = Field(default_factory=lambda: int(os.getenv("CAMERA_PC_INDEX", os.getenv("CAMERA_INDEX", "0"))))
    width: int = Field(default_factory=lambda: int(os.getenv("CAMERA_PC_WIDTH", os.getenv("CAMERA_WIDTH", "1280"))))
    height: int = Field(default_factory=lambda: int(os.getenv("CAMERA_PC_HEIGHT", os.getenv("CAMERA_HEIGHT", "720"))))
    fps: int = Field(default_factory=lambda: int(os.getenv("CAMERA_PC_FPS", os.getenv("CAMERA_FPS", "30"))))

class CameraSettings(BaseModel):
    source: str = Field(default_factory=lambda: os.getenv("CAMERA_SOURCE", "pc"))
    index: int = Field(default_factory=lambda: int(os.getenv("CAMERA_INDEX", "0")))
    laptop_index: int = Field(default_factory=lambda: int(os.getenv("CAMERA_LAPTOP_INDEX", "0")))
    smart_board_index: int = Field(default_factory=lambda: int(os.getenv("CAMERA_SMART_BOARD_INDEX", "1")))
    external_index: int = Field(default_factory=lambda: int(os.getenv("CAMERA_EXTERNAL_INDEX", "2")))
    extension_index: int = Field(default_factory=lambda: int(os.getenv("CAMERA_EXTENSION_INDEX", "1")))
    width: int = Field(default_factory=lambda: int(os.getenv("CAMERA_WIDTH", "1280")))
    height: int = Field(default_factory=lambda: int(os.getenv("CAMERA_HEIGHT", "720")))
    fps: int = Field(default_factory=lambda: int(os.getenv("CAMERA_FPS", "30")))
    pc: PCCameraSettings = Field(default_factory=PCCameraSettings)
    droidcam: DroidCamSettings = Field(default_factory=DroidCamSettings)
    esp32: ESP32CamSettings = Field(default_factory=ESP32CamSettings)
    retry: CameraRetrySettings = Field(default_factory=CameraRetrySettings)

class DetectionSettings(BaseModel):
    model_type: str = Field(default_factory=lambda: os.getenv("DETECTION_MODEL_TYPE", "scrfd")) # "scrfd" or "yolov8"
    model_name: str = Field(default_factory=lambda: os.getenv("DETECTOR_MODEL", "models/face_detection/scrfd/scrfd_10g_kps.onnx"))
    model_path: str = Field(default_factory=lambda: os.getenv("DETECTION_MODEL_PATH", "models/face_detection/scrfd/scrfd_10g_kps.onnx"))
    input_width: int = Field(default_factory=lambda: int(os.getenv("DETECTION_INPUT_WIDTH", "640")))
    input_height: int = Field(default_factory=lambda: int(os.getenv("DETECTION_INPUT_HEIGHT", "640")))
    input_size: int = Field(default_factory=lambda: int(os.getenv("DETECTION_INPUT_SIZE", "640")))
    confidence_threshold: float = Field(default_factory=lambda: float(os.getenv("DETECTION_CONFIDENCE", "0.35")))
    nms_iou_threshold: float = Field(default_factory=lambda: float(os.getenv("DETECTION_NMS_IOU", "0.40")))
    padding_ratio: float = Field(default_factory=lambda: float(os.getenv("DETECTION_PADDING", "0.0")))
    max_faces: int = Field(default_factory=lambda: int(os.getenv("DETECTION_MAX_FACES", "100")))
    min_face_size: int = Field(default_factory=lambda: int(os.getenv("DETECTION_MIN_FACE_SIZE", "16")))
    enable_tiling: bool = Field(default_factory=lambda: os.getenv("DETECTION_ENABLE_TILING", "True").lower() == "true")
    tile_interval: int = Field(default_factory=lambda: int(os.getenv("DETECTION_TILE_INTERVAL", "3")))
    debug_mode: bool = Field(default_factory=lambda: os.getenv("DETECTION_DEBUG_MODE", "False").lower() == "true")

class QualitySettings(BaseModel):
    min_face_area: int = Field(default_factory=lambda: int(os.getenv("MIN_FACE_AREA", "3000"))) # FAR zone minimum (~55x55px)
    min_face_dimension: int = Field(default_factory=lambda: int(os.getenv("MIN_FACE_DIMENSION", "36")))
    min_sharpness: float = Field(default_factory=lambda: float(os.getenv("MIN_SHARPNESS", "35.0"))) # Laplacian variance threshold
    min_brightness: float = Field(default_factory=lambda: float(os.getenv("MIN_BRIGHTNESS", "30.0")))
    max_brightness: float = Field(default_factory=lambda: float(os.getenv("MAX_BRIGHTNESS", "230.0")))
    min_contrast: float = Field(default_factory=lambda: float(os.getenv("MIN_CONTRAST", "12.0")))
    max_glare_ratio: float = Field(default_factory=lambda: float(os.getenv("MAX_GLARE_RATIO", "0.40")))
    max_tilt_angle: float = Field(default_factory=lambda: float(os.getenv("MAX_TILT_ANGLE", "35.0")))
    max_yaw_offset: float = Field(default_factory=lambda: float(os.getenv("MAX_YAW_OFFSET", "0.80")))

class RecognitionSettings(BaseModel):
    model_name: str = Field(default_factory=lambda: os.getenv("RECOGNITION_MODEL", "w600k_mbf.onnx"))
    embedding_dim: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "512")))
    similarity_threshold: float = Field(default_factory=lambda: float(os.getenv("RECOGNITION_THRESHOLD", "0.50")))
    margin_threshold: float = Field(default_factory=lambda: float(os.getenv("MARGIN_THRESHOLD", "0.08")))
    top_k: int = Field(default_factory=lambda: int(os.getenv("RECOGNITION_TOP_K", "10")))
    index_path: str = Field(default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "database/faiss_index.bin"))
    device: str = Field(default_factory=lambda: os.getenv("RECOGNITION_DEVICE", "cpu"))

class TrackingSettings(BaseModel):
    track_high_thresh: float = Field(default_factory=lambda: float(os.getenv("TRACK_HIGH_THRESH", "0.5")))
    track_low_thresh: float = Field(default_factory=lambda: float(os.getenv("TRACK_LOW_THRESH", "0.1")))
    new_track_thresh: float = Field(default_factory=lambda: float(os.getenv("NEW_TRACK_THRESH", "0.6")))
    match_thresh: float = Field(default_factory=lambda: float(os.getenv("TRACK_MATCH_THRESH", "0.7"))) # 1 - IoU distance (0.7 -> IoU >= 0.3)
    max_lost_frames: int = Field(default_factory=lambda: int(os.getenv("MAX_LOST_FRAMES", "30")))
    min_hits_to_activate: int = Field(default_factory=lambda: int(os.getenv("MIN_HITS_TO_ACTIVATE", "2")))
    fps: int = Field(default_factory=lambda: int(os.getenv("TRACKING_FPS", "30")))

class StabilizationSettings(BaseModel):
    history_length: int = Field(default_factory=lambda: int(os.getenv("STABILIZATION_HISTORY_LEN", "15")))
    min_stable_observations: int = Field(default_factory=lambda: int(os.getenv("MIN_STABLE_OBSERVATIONS", "3")))
    identity_switch_threshold: int = Field(default_factory=lambda: int(os.getenv("IDENTITY_SWITCH_THRESHOLD", "4")))
    unknown_persistence_duration: int = Field(default_factory=lambda: int(os.getenv("UNKNOWN_PERSISTENCE_DURATION", "5")))
    poor_quality_timeout: int = Field(default_factory=lambda: int(os.getenv("POOR_QUALITY_TIMEOUT", "8")))
    recognition_interval: int = Field(default_factory=lambda: int(os.getenv("RECOGNITION_INTERVAL", "5")))
    re_embedding_on_low_confidence: bool = Field(default_factory=lambda: os.getenv("RE_EMBED_LOW_CONF", "True").lower() == "true")

class SessionSettings(BaseModel):
    default_duration_minutes: int = Field(default_factory=lambda: int(os.getenv("SESSION_DURATION_MINUTES", "45")))
    pre_session_window_minutes: int = Field(default_factory=lambda: int(os.getenv("PRE_SESSION_WINDOW_MINUTES", "15")))
    post_session_window_minutes: int = Field(default_factory=lambda: int(os.getenv("POST_SESSION_WINDOW_MINUTES", "10")))
    auto_end_on_expiry: bool = Field(default_factory=lambda: os.getenv("SESSION_AUTO_END", "True").lower() == "true")
    timezone: str = Field(default_factory=lambda: os.getenv("TIMEZONE", "Asia/Kolkata"))

class AttendanceSettings(BaseModel):
    late_threshold_minutes: float = Field(default_factory=lambda: float(os.getenv("LATE_THRESHOLD_MINUTES", "10.0")))
    allow_pre_session_marking: bool = Field(default_factory=lambda: os.getenv("ALLOW_PRE_SESSION_MARKING", "True").lower() == "true")
    min_similarity_threshold: float = Field(default_factory=lambda: float(os.getenv("ATTENDANCE_MIN_SIMILARITY", "0.65")))
    require_phase7_confirmed: bool = Field(default_factory=lambda: os.getenv("REQUIRE_PHASE7_CONFIRMED", "True").lower() == "true")
    deduplication_buffer_seconds: float = Field(default_factory=lambda: float(os.getenv("DEDUP_BUFFER_SECONDS", "1.0")))

class DashboardSettings(BaseModel):
    host: str = Field(default_factory=lambda: os.getenv("DASHBOARD_HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("DASHBOARD_PORT", "8000")))
    cors_origins: str = Field(default_factory=lambda: os.getenv("DASHBOARD_CORS_ORIGINS", "*"))
    poll_interval_ms: int = Field(default_factory=lambda: int(os.getenv("DASHBOARD_POLL_INTERVAL_MS", "1500")))
    debug_mode_default: bool = Field(default_factory=lambda: os.getenv("DASHBOARD_DEBUG_MODE", "False").lower() == "true")

class AppSettings(BaseModel):
    environment: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_dir: str = Field(default_factory=lambda: os.getenv("LOG_DIR", "logs"))
    db_path: str = Field(default_factory=lambda: os.getenv("DB_PATH", "database/smartclass.sqlite"))
    camera: CameraSettings = Field(default_factory=CameraSettings)
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    recognition: RecognitionSettings = Field(default_factory=RecognitionSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    stabilization: StabilizationSettings = Field(default_factory=StabilizationSettings)
    session: SessionSettings = Field(default_factory=SessionSettings)
    attendance: AttendanceSettings = Field(default_factory=AttendanceSettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)

def get_settings() -> AppSettings:
    """Returns the centralized application settings, incorporating optional camera.yaml if present."""
    settings = AppSettings()
    config_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_paths = [
        os.path.join(config_dir, "camera.yaml"),
        os.path.join(config_dir, "camera_config.yaml"),
    ]
    for yp in yaml_paths:
        if os.path.exists(yp):
            try:
                import yaml
                with open(yp, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if data and isinstance(data, dict):
                    cam_data = data.get("camera", {})
                    if "source" in cam_data:
                        settings.camera.source = str(cam_data["source"])
                    if "pc" in cam_data and isinstance(cam_data["pc"], dict):
                        pc_conf = cam_data["pc"]
                        if "device_index" in pc_conf:
                            dev_idx = int(pc_conf["device_index"])
                            settings.camera.pc.device_index = dev_idx
                            settings.camera.laptop_index = dev_idx
                            settings.camera.index = dev_idx
                        if "width" in pc_conf:
                            w = int(pc_conf["width"])
                            settings.camera.pc.width = w
                            settings.camera.width = w
                        if "height" in pc_conf:
                            h = int(pc_conf["height"])
                            settings.camera.pc.height = h
                            settings.camera.height = h
                        if "fps" in pc_conf:
                            fps = int(pc_conf["fps"])
                            settings.camera.pc.fps = fps
                            settings.camera.fps = fps
            except Exception:
                pass
            break
    return settings


