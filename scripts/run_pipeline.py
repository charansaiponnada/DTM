from env_policy import ensure_running_in_conda_env
from pipeline import main


if __name__ == "__main__":
    ensure_running_in_conda_env()
    main()
