from click.testing import CliRunner
from src.cli import main


def test_cli_help_succeeds():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "DTM Drainage AI Pipeline" in result.output
