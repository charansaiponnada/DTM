from env_policy import ensure_running_in_workspace_venv
from pipeline import main


if __name__ == "__main__":
    ensure_running_in_workspace_venv()
    main()
