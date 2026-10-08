"""Regenerate all public analyses, tables, and paper figures.

Industry aggregates are committed but cannot be regenerated without private data.
"""

from acceptance import run as acceptance
from main_analysis import run as main_analysis
from mechanism import run as mechanism
from mitigation import run as mitigation
from robustness import run as robustness
from validate import main as validate


def main() -> None:
    """Run every analysis that uses public data."""
    main_analysis()
    acceptance()
    robustness()
    mechanism()
    mitigation()
    validate()
    print("Reproduction complete. See results/ and figures/.")


if __name__ == "__main__":
    main()
