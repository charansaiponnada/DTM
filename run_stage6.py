"""Run Stage 6 (Drainage Network Design) on existing DTM + hydro outputs."""
from src.pipeline import DTMDrainagePipeline

pipeline = DTMDrainagePipeline(
    config_path="config/config.yaml",
    output_dir="data/output",
)
pipeline.stage6_drainage_design()
pipeline.run_evaluation()
