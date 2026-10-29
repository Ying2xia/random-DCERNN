"""Numerical refinement and directional communication validation."""

from __future__ import annotations

import pathlib
import sys
import unittest

CODE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from validate_refinement import run_refinement_validation  # noqa: E402


class RefinementValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validation = run_refinement_validation()

    def test_more_refinement_rounds_reduce_centralized_prediction_gap(self) -> None:
        distances = self.validation.prediction_distances
        ordered = [distances[rounds] for rounds in (0, 1, 2, 3, 5)]
        improving_transitions = sum(
            next_value < current
            for current, next_value in zip(ordered, ordered[1:])
        )

        self.assertTrue(all(value >= 0.0 for value in ordered))
        self.assertGreaterEqual(
            improving_transitions,
            3,
            msg=f"Refinement distance trajectory was {distances}.",
        )
        self.assertLess(
            distances[5],
            0.85 * distances[0],
            msg=f"Refinement distance trajectory was {distances}.",
        )
        self.assertLess(
            sum(ordered[1:]) / 4.0,
            0.9 * ordered[0],
            msg=f"Refinement distance trajectory was {distances}.",
        )

    def test_every_expansion_point_matches_centralized_gradient(self) -> None:
        for rounds, errors in self.validation.gradient_match_errors.items():
            with self.subTest(rounds=rounds):
                self.assertEqual(len(errors), rounds)
                self.assertTrue(
                    all(error < 1e-10 for error in errors),
                    msg=f"Gradient match errors for T={rounds}: {errors}.",
                )

    def test_communication_counts_only_two_remote_message_directions(self) -> None:
        per_round = self.validation.expected_bytes_per_round
        for rounds, actual in self.validation.communication_bytes.items():
            with self.subTest(rounds=rounds):
                self.assertEqual(actual, rounds * per_round)

        # The formula is exactly one host-to-remote parameter broadcast and one
        # remote-to-host gradient upload.  The host's local copies are excluded.
        self.assertEqual(self.validation.communication_bytes[0], 0)
        self.assertGreater(per_round, 0)


if __name__ == "__main__":
    unittest.main()
