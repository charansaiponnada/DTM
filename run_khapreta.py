"""Run DTM pipeline for KHAPRETA village only."""
from src.pipeline import DTMDrainagePipeline

pipeline = DTMDrainagePipeline(
    config_path="config/config.yaml",
    input_las="data/input/Gujrat_Point_Cloud/KHAPRETA_510206.laz",
    output_dir="data/output/KHAPRETA",
)
pipeline.run(use_ml_refine=False, stream_threshold=1000)
pipeline.run_evaluation()
