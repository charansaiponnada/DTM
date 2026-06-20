"""Re-run Stages 4 (Hydrology) and 6 (Drainage Design) on existing DTM."""
from src.pipeline import DTMDrainagePipeline

pipeline = DTMDrainagePipeline(
    config_path="config/config.yaml",
    output_dir="data/output",
)
pipeline.stage4_hydrology(stream_threshold=1000)
pipeline.stage5_waterlogging()
pipeline.stage6_drainage_design()
pipeline.run_evaluation()
