# Activity Log

## 2026-02-27

- 2026-02-27T10:20:00 | CREATED | Initial preflight scanner and baseline stage scripts.
- 2026-02-27T10:45:00 | FAILED | First full run failed due zero DTM outputs.
- 2026-02-27T10:50:00 | FIXED | Added fallback ground selection for class-0-only data.
- 2026-02-27T10:55:00 | SUCCESS | End-to-end baseline run succeeded.
- 2026-02-27T11:10:00 | CREATED | ML training and optimization scripts added.
- 2026-02-27T11:30:00 | SUCCESS | Optimization outputs generated.
- 2026-02-27T12:34:00 | UPDATED | Stage 1 enhanced with SOR/ROR noise filtering.
- 2026-02-27T12:44:00 | FIXED | Noise filtering fallback added to prevent over-pruning.
- 2026-02-27T14:05:00 | UPDATED | Ground filtering migrated to `laspy + scipy` with WhiteboxTools-assisted mode.
- 2026-02-27T14:35:00 | UPDATED | Full docs rewrite completed for clarity and strict `.venv + requirements.txt` policy.
