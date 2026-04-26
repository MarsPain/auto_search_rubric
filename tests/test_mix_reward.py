from __future__ import annotations

import unittest

from autosr.mix_reward import EtaScheduler, blended_reward


class MixRewardTest(unittest.TestCase):
    def test_blended_reward_clamps_eta_to_reward_endpoints(self) -> None:
        self.assertEqual(2.0, blended_reward(2.0, 10.0, -0.5))
        self.assertEqual(10.0, blended_reward(2.0, 10.0, 1.5))

    def test_blended_reward_interpolates_between_rewards(self) -> None:
        self.assertEqual(6.0, blended_reward(2.0, 10.0, 0.5))

    def test_eta_scheduler_holds_start_through_warmup(self) -> None:
        scheduler = EtaScheduler(warmup_steps=3, ramp_steps=10, start=0.2, end=0.8)

        self.assertEqual(0.2, scheduler.value(0))
        self.assertEqual(0.2, scheduler.value(3))

    def test_eta_scheduler_ramps_linearly_after_warmup(self) -> None:
        scheduler = EtaScheduler(warmup_steps=2, ramp_steps=4, start=0.1, end=0.9)

        self.assertAlmostEqual(0.5, scheduler.value(4))

    def test_eta_scheduler_clamps_to_end_after_ramp(self) -> None:
        scheduler = EtaScheduler(warmup_steps=2, ramp_steps=4, start=0.1, end=0.9)

        self.assertEqual(0.9, scheduler.value(10))

    def test_eta_scheduler_uses_end_when_ramp_is_disabled_after_warmup(self) -> None:
        scheduler = EtaScheduler(warmup_steps=2, ramp_steps=0, start=0.1, end=0.9)

        self.assertEqual(0.1, scheduler.value(2))
        self.assertEqual(0.9, scheduler.value(3))


if __name__ == "__main__":
    unittest.main()
