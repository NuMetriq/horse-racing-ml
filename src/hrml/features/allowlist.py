from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


# Columns that are never features (ids, labels, post-race outcomes, etc.)
DEFAULT_BLOCKLIST: set[str] = {
    # identifiers / join keys
    "race_id",
    "horse_id",
    "jockey_id",
    "trainer_id",
    "race_date",
    "race_date_ord",
    # targets / outcomes
    "is_winner",
    "finish_position",
    "finish_time",
    "place",  # if it exists later
    "position",  # generic risky names
    # anything with obvious leakage semantics (expand as needed)
    "winner",
    "won",
    "starting_price",  # if you ever have it
    "sp",  # generic
    "field_size_r",
    "sp_decimal",
    "odds_implied",
    "odds_implied_norm",
}

# If these appear as numeric columns and aren't explicitly allowed/prefixed, we fail.
DEFAULT_LEAKAGE_PATTERNS: tuple[str, ...] = (
    "finish",
    "finish_",
    "finishposition",
    "finish_position",
    "place",
    "placing",
    "time",
    "result",
    "winner",
    "won",
    "payout",
    "dividend",
)


@dataclass(frozen=True)
class FeatureAllowlist:
    """
    Feature selection policy:
      - explicit allowlist for race/context features
      - prefix rules for engineered features (horse_, jockey_, trainer_, ...)
      - blocklist for identifiers and labels
      - strict mode: fail if unknown numeric columns appear
    """
    explicit: tuple[str, ...] = (
        # race-level context (safe pre-race fields)
        "dist_f",
        "field_size",
        "race_class",
        # odds baseline
        #"odds_implied",
        #"odds_implied_norm",
        # add others only if you are sure they're pre-race
        # e.g. "going" if encoded safely downstream
        "post_position",
        "weight_lbs",
        "or_rating",
        "rpr_rating",
        "ts_rating",
    )
    prefixes: tuple[str, ...] = (
        "horse_",
        "jockey_",
        "trainer_",
        "jockey_trainer_",
    )
    blocklist: frozenset[str] = field(default_factory=lambda: frozenset(DEFAULT_BLOCKLIST))
    strict_unknown_numeric: bool = True
    leakage_token_block: tuple[str, ...] = DEFAULT_LEAKAGE_PATTERNS

    def select(self, df: pd.DataFrame) -> List[str]:
        # Candidate columns: explicit + prefix matches
        cols = []
        for c in df.columns:
            if c in self.blocklist:
                continue
            if c in self.explicit:
                cols.append(c)
                continue
            if any(c.startswith(p) for p in self.prefixes):
                cols.append(c)
                continue

        # Enforce deterministic ordering
        cols = sorted(set(cols))

        # Strict check: if df has numeric columns not accounted for, fail loudly
        if self.strict_unknown_numeric:
            numeric_cols = set(df.select_dtypes(include=["number", "bool"]).columns)
            allowed = set(cols) | set(self.blocklist)

            unknown_numeric = sorted(numeric_cols - allowed)

            # Filter out safe-ish “derived” numeric columns you intentionally keep elsewhere
            # (prefer to add them to explicit rather than weaken this check)

            if unknown_numeric:
                # If any unknown numeric column smells like post-race leakage, be explicit
                bad = [c for c in unknown_numeric if any(tok in c.lower() for tok in self.leakage_token_block)]
                msg = [
                    "Unknown numeric columns detected (not in allowlist/prefix rules).",
                    "This is a CI-style failure to prevent silent leakage regressions.",
                    "",
                    "Unknown numeric columns:",
                    *[f"  - {c}" for c in unknown_numeric],
                ]
                if bad:
                    msg += [
                        "",
                        "These look like potential leakage/outcome fields:",
                        *[f"  - {c}" for c in bad],
                    ]
                raise AssertionError("\n".join(msg))

        return cols


def write_feature_manifest(feature_cols: Sequence[str], out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_cols = list(feature_cols)
    feature_cols_sorted = sorted(feature_cols)

    sha = hashlib.sha256(("\n".join(feature_cols_sorted)).encode("utf-8")).hexdigest()

    payload = {
        "n_features": len(feature_cols_sorted),
        "sha256": sha,
        "features": feature_cols_sorted,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload