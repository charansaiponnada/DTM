"""Run evaluation on KHAPRETA outputs."""
from src.pipeline import DTMDrainagePipeline

pipeline = DTMDrainagePipeline(
    config_path="config/config.yaml",
    output_dir="data/output/KHAPRETA",
)
try:
    pipeline.run_evaluation()
except Exception as exc:
    print(f"Evaluation warning (non-fatal): {exc}")
