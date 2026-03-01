# Documentation Protocol

Use this protocol for every change.

## Before coding
1. Define stage scope.
2. Define expected outputs.
3. Confirm `.venv` is active.

## During coding
1. Keep edits stage-specific.
2. Record failures and fixes in `logs/activity_log.md`.
3. Avoid undocumented command changes.

## After coding
1. Run validation command(s).
2. Update relevant docs (`README`, `newbie`, `final`, stage docs).
3. Add timestamped log entries.

## Mandatory environment rule
- Never run scripts with global Python.
- Always use `.venv\Scripts\python.exe`.
- Install dependencies only through `requirements.txt`.

## Log entry format
`YYYY-MM-DDTHH:MM:SS | STATUS | concise action/result`

Statuses:
- `CREATED`
- `UPDATED`
- `FAILED`
- `FIXED`
- `SUCCESS`
