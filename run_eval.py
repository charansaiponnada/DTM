"""Run evaluation on completed pipeline outputs."""
from src.pipeline import DTMDrainagePipeline
import sys, traceback

pipeline = DTMDrainagePipeline(
    config_path="config/config.yaml",
    output_dir="data/output",
)
try:
    results = pipeline.run_evaluation()
    import json
    print(json.dumps(results, indent=2, default=str))
except Exception as exc:
    traceback.print_exc()
    sys.exit(1)
