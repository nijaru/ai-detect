"""Consensus voting logic for ensemble detection."""

from dataclasses import dataclass
from enum import Enum

from .models import DetectionResult


class Agreement(Enum):
    """Level of agreement among models."""

    UNANIMOUS = "unanimous"
    STRONG = "strong"  # 4/5 or 80%+
    MAJORITY = "majority"  # >50%
    SPLIT = "split"  # 50/50


@dataclass
class ConsensusResult:
    """Aggregated result from all detectors."""

    is_ai: bool
    confidence: float
    agreement: Agreement
    votes_ai: int
    votes_real: int
    individual_results: list[DetectionResult]

    @property
    def total_votes(self) -> int:
        return self.votes_ai + self.votes_real

    @property
    def agreement_ratio(self) -> float:
        if self.total_votes == 0:
            return 0.0
        winning_votes = self.votes_ai if self.is_ai else self.votes_real
        return winning_votes / self.total_votes


def compute_consensus(
    results: list[DetectionResult],
    threshold: float = 0.5,
) -> ConsensusResult:
    """Compute consensus verdict from multiple detector results.

    Uses confidence-weighted voting where each model's vote is weighted
    by its confidence score.
    """
    if not results:
        return ConsensusResult(
            is_ai=False,
            confidence=0.0,
            agreement=Agreement.SPLIT,
            votes_ai=0,
            votes_real=0,
            individual_results=[],
        )

    weighted_ai = 0.0
    weighted_real = 0.0
    votes_ai = 0
    votes_real = 0

    for result in results:
        weight = result.confidence
        if result.is_ai:
            weighted_ai += weight
            votes_ai += 1
        else:
            weighted_real += weight
            votes_real += 1

    total_weight = weighted_ai + weighted_real
    if total_weight == 0:
        ai_probability = 0.5
    else:
        ai_probability = weighted_ai / total_weight

    is_ai = ai_probability >= threshold

    if is_ai:
        confidence = ai_probability
    else:
        confidence = 1 - ai_probability

    total_votes = len(results)
    winning_votes = votes_ai if is_ai else votes_real

    if winning_votes == total_votes:
        agreement = Agreement.UNANIMOUS
    elif winning_votes / total_votes >= 0.8:
        agreement = Agreement.STRONG
    elif winning_votes / total_votes > 0.5:
        agreement = Agreement.MAJORITY
    else:
        agreement = Agreement.SPLIT

    return ConsensusResult(
        is_ai=is_ai,
        confidence=confidence,
        agreement=agreement,
        votes_ai=votes_ai,
        votes_real=votes_real,
        individual_results=results,
    )
