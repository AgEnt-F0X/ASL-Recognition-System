import unittest

import numpy as np

from gesture_text import GestureTextComposer, TemporalPredictionFilter


class GestureTextComposerTests(unittest.TestCase):
    def test_hello_can_include_a_deliberate_double_l(self):
        composer = GestureTextComposer(hold_seconds=0.5, release_seconds=0.2)
        committed = []

        def hold(character, start):
            committed.append(composer.update(character, True, start).committed_character)
            committed.append(composer.update(character, True, start + 0.5).committed_character)

        hold("H", 0.0)
        hold("E", 0.6)
        hold("L", 1.2)
        self.assertIsNone(composer.update("L", True, 1.8).committed_character)
        composer.update(None, False, 1.9)
        composer.update(None, False, 2.2)
        hold("L", 2.3)
        hold("O", 2.9)

        self.assertEqual("".join(character for character in committed if character), "HELLO")

    def test_same_held_sign_is_not_repeated(self):
        composer = GestureTextComposer(hold_seconds=0.5, release_seconds=0.2)
        self.assertIsNone(composer.update("A", True, 0.0).committed_character)
        self.assertEqual(composer.update("A", True, 0.5).committed_character, "A")
        self.assertIsNone(composer.update("A", True, 1.5).committed_character)


class TemporalPredictionFilterTests(unittest.TestCase):
    def test_consistent_probabilities_become_stable(self):
        prediction_filter = TemporalPredictionFilter(history_length=6, minimum_frames=4)
        probabilities = np.array([0.02, 0.03, 0.90, 0.05], dtype=np.float32)

        for _ in range(3):
            self.assertFalse(prediction_filter.update(probabilities).is_stable)

        result = prediction_filter.update(probabilities)
        self.assertTrue(result.is_stable)
        self.assertEqual(result.label_index, 2)

    def test_ambiguous_probabilities_are_not_stable(self):
        prediction_filter = TemporalPredictionFilter(history_length=6, minimum_frames=4)
        probabilities = np.array([0.45, 0.43, 0.07, 0.05], dtype=np.float32)

        for _ in range(6):
            result = prediction_filter.update(probabilities)

        self.assertFalse(result.is_stable)

    def test_filter_and_composer_can_enter_hello(self):
        prediction_filter = TemporalPredictionFilter(history_length=6, minimum_frames=4)
        composer = GestureTextComposer(hold_seconds=0.2, release_seconds=0.2)
        committed = []
        timestamp = 0.0

        def feed_letter(letter_index):
            nonlocal timestamp
            probabilities = np.full(26, 0.002, dtype=np.float32)
            probabilities[letter_index] = 0.95
            probabilities /= probabilities.sum()
            for _ in range(12):
                result = prediction_filter.update(probabilities)
                character = chr(65 + result.label_index) if result.is_stable else None
                update = composer.update(character, True, timestamp)
                if update.committed_character:
                    committed.append(update.committed_character)
                timestamp += 0.1

        for letter_index in [7, 4, 11]:
            feed_letter(letter_index)

        for _ in range(4):
            prediction_filter.update(None)
            composer.update(None, False, timestamp)
            timestamp += 0.1

        for letter_index in [11, 14]:
            feed_letter(letter_index)

        self.assertEqual("".join(committed), "HELLO")


if __name__ == "__main__":
    unittest.main()
