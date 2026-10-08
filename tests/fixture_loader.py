from pathlib import Path


_FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> str:
    return (_FIXTURE_DIRECTORY / name).read_text(encoding="utf-8")
