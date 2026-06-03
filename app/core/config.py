from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # GCP
    gcp_project_id: str
    gcp_region: str = "us-central1"

    # Cloud Storage
    gcs_bucket_name: str
    gcs_raw_prefix: str = "raw/"
    gcs_results_prefix: str = "results/"
    gcs_checkpoints_prefix: str = "checkpoints/"

    # BigQuery
    bq_dataset: str = "palimpsest_results"
    bq_table: str = "change_detections"

    # Artifact Registry
    ar_repository: str = "palimpsest-repo"
    ar_image_name: str = "palimpsest-api"

    # Earth Engine
    ee_project: str

    # Model
    model_checkpoint_path: str = "checkpoints/changeformer_oscd.pth"
    # Threshold calibrated on OSCD val set (nantes/mumbai/bordeaux) after
    # fine-tuning on Sentinel-2 10 m/px imagery.  Sweep peak F1=0.2932@0.24
    # but 0.18 gives F1=0.2915 with better real-world recall (less false negatives).
    confidence_threshold: float = 0.18
    patch_size: int = 512
    tta_enabled: bool = True
    # Min connected-component size to keep after thresholding.
    # At 10 m/px: 5 px = 500 m² (was 50 px = 5 000 m², too aggressive).
    min_change_area_px: int = 5

    # Gemini — Google AI Studio key (aistudio.google.com/app/apikey)
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    # API
    api_port: int = 8080
    log_level: str = "INFO"


settings = Settings()
