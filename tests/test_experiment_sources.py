from pathlib import Path


def test_experiment_scripts_are_valid_python():
    project_root = Path(__file__).parents[1]
    scripts = sorted((project_root / "experiments").glob("*.py"))

    assert scripts
    for script in scripts:
        compile(script.read_text(encoding="utf-8"), str(script), "exec")
