# Documentation Protocol (Mandatory)

This project follows a strict documentation-first workflow for every change.

## 1) Before Implementation

1. Add or update TODO items in planning notes.
2. Record scope and expected outputs in relevant docs.
3. Confirm target stage and acceptance criteria.
4. Confirm execution interpreter is from active Conda env.

## 2) During Implementation

1. Keep changes stage-scoped (do not mix unrelated edits).
2. Track each meaningful step in `logs/activity_log.md` with timestamp and status.
3. Use statuses: `CREATED`, `UPDATED`, `FAILED`, `FIXED`, `SUCCESS`.

## 3) After Implementation

1. Validate run or output generation.
2. Update usage instructions if commands changed.
3. Update progress summary in `newbie.md` if project status changed.
4. Update completion expectations in `final.md` if deliverables changed.

## 4) Logging Format

Use one line per event:

`YYYY-MM-DDTHH:MM:SS | STATUS | short message with file/output link`

Example:

`2026-02-27T12:10:00 | SUCCESS | Hydrology export completed with outputs in outputs/hydrology/`

## 5) Branching Rule for This Hackathon Repo

- Keep workflow lightweight and direct.
- Prefer single-branch iterative development with frequent small verified updates.
- Prioritize traceability over complexity.

## 6) Environment Rule (Mandatory)

- Never run project scripts with global/system Python.
- Every execution must use `python` from active `conda` env.
- Every dependency change must be applied through `environment.yml`.
- Preferred for this hackathon: `conda activate dataset-dtm` (from `environment.yml`).
