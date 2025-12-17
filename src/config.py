from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def reports_figures(self) -> Path:
        return self.root / "reports" / "figures"

    @property
    def reports_tables(self) -> Path:
        return self.root / "reports" / "tables"


def get_paths() -> Paths:
    # `src/` is one level below repo root
    root = Path(__file__).resolve().parents[1]
    return Paths(root=root)


# GSS Data Config
REQUIRED_COLUMNS: tuple[str, ...] = ("year", "happy")


def choose_default_gss_path(repo_root: Path) -> Path:
    """Default to gss_extract.csv if present."""
    return repo_root / "data" / "raw" / "gss_extract.csv"
