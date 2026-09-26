"""Temporal filtering and deliberate letter entry for the ASL camera app."""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence
import time

import numpy as np


@dataclass(frozen=True)
class SmoothedPrediction:
    label_index: Optional[int]
    confidence: float = 0.0
    margin: float = 0.0

    @property
    def is_stable(self) -> bool:
        return self.label_index is not None


class TemporalPredictionFilter:
    """Accept only consistent, sufficiently separated model predictions."""

    def __init__(
        self,
        history_length: int = 10,
        minimum_frames: int = 6,
        minimum_frame_confidence: float = 0.50,
        minimum_confidence: float = 0.68,
        minimum_margin: float = 0.10,
        minimum_vote_ratio: float = 0.70,
    ):
        self.history = deque(maxlen=history_length)
        self.minimum_frames = minimum_frames
        self.minimum_frame_confidence = minimum_frame_confidence
        self.minimum_confidence = minimum_confidence
        self.minimum_margin = minimum_margin
        self.minimum_vote_ratio = minimum_vote_ratio
        self.missed_frames = 0

    def reset(self) -> None:
        self.history.clear()
        self.missed_frames = 0

    def update(self, probabilities: Optional[Sequence[float]]) -> SmoothedPrediction:
        if probabilities is None:
            self.missed_frames += 1
            if self.missed_frames >= 4:
                self.reset()
            return SmoothedPrediction(None)

        values = np.asarray(probabilities, dtype=np.float32)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("Expected a one-dimensional probability vector.")

        top_two = np.partition(values, -2)[-2:]
        raw_confidence = float(top_two[1])
        if raw_confidence < self.minimum_frame_confidence:
            self.missed_frames += 1
            if self.missed_frames >= 4:
                self.reset()
            return SmoothedPrediction(None)

        self.missed_frames = 0
        self.history.append(values)
        if len(self.history) < self.minimum_frames:
            return SmoothedPrediction(None)

        averaged = np.mean(np.stack(self.history), axis=0)
        label_index = int(np.argmax(averaged))
        top_two = np.partition(averaged, -2)[-2:]
        confidence = float(top_two[1])
        margin = float(top_two[1] - top_two[0])
        vote_ratio = sum(int(np.argmax(sample)) == label_index for sample in self.history) / len(self.history)

        if (
            confidence < self.minimum_confidence
            or margin < self.minimum_margin
            or vote_ratio < self.minimum_vote_ratio
        ):
            return SmoothedPrediction(None, confidence, margin)

        return SmoothedPrediction(label_index, confidence, margin)


@dataclass(frozen=True)
class EntryUpdate:
    committed_character: Optional[str]
    progress: float
    message: str


class GestureTextComposer:
    """Commit one stable gesture at a time and safely handle double letters.

    A different sign can follow immediately. To repeat the same sign (the two
    Ls in HELLO, for example), the user briefly removes their hand first.
    """

    def __init__(self, hold_seconds: float = 0.80, release_seconds: float = 0.35):
        self.hold_seconds = hold_seconds
        self.release_seconds = release_seconds
        self.last_committed_character: Optional[str] = None
        self.candidate_character: Optional[str] = None
        self.candidate_started_at: Optional[float] = None
        self.no_hand_started_at: Optional[float] = None

    def reset(self) -> None:
        self.last_committed_character = None
        self.candidate_character = None
        self.candidate_started_at = None
        self.no_hand_started_at = None

    def update(
        self,
        stable_character: Optional[str],
        hand_detected: bool,
        now: Optional[float] = None,
    ) -> EntryUpdate:
        now = time.monotonic() if now is None else now

        if not hand_detected:
            if self.no_hand_started_at is None:
                self.no_hand_started_at = now
            self._clear_candidate()
            if (
                self.last_committed_character is not None
                and now - self.no_hand_started_at >= self.release_seconds
            ):
                self.last_committed_character = None
                return EntryUpdate(None, 0.0, "Ready for the next letter")
            return EntryUpdate(None, 0.0, "Hand not detected")

        self.no_hand_started_at = None
        if stable_character is None:
            self._clear_candidate()
            return EntryUpdate(None, 0.0, "Hold one clear hand sign")

        if stable_character == self.last_committed_character:
            self._clear_candidate()
            return EntryUpdate(None, 0.0, "Letter added - lower hand briefly to repeat it")

        if stable_character != self.candidate_character:
            self.candidate_character = stable_character
            self.candidate_started_at = now
            return EntryUpdate(None, 0.0, "Hold sign steady")

        candidate_started_at = self.candidate_started_at
        if candidate_started_at is None:
            return EntryUpdate(None, 0.0, "Hold sign steady")

        duration = now - candidate_started_at
        progress = min(1.0, duration / self.hold_seconds)
        if duration < self.hold_seconds:
            return EntryUpdate(None, progress, "Hold sign steady")

        committed_character = stable_character
        self.last_committed_character = committed_character
        self._clear_candidate()
        return EntryUpdate(committed_character, 1.0, f"Added {committed_character}")

    def _clear_candidate(self) -> None:
        self.candidate_character = None
        self.candidate_started_at = None
