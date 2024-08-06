import unittest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from agent.dqn.utils import (
    ReplayBuffer,
    select_action,
    update_epsilon,
    plot_results,
    console,
    save_final_results,
    find_non_masked_rows,
    compute_loss_mm,
    compute_loss_explore,
    update_model,
    select_action,
    save_validation,
    save_states_q_values_actions,
    target_hard_update,
    update_epsilon,
)


class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer = ReplayBuffer(size=100, batch_size=32)

    def test_store(self):
        obs = {"state": 1}
        act_explore = 2
        act_mm = [0, 1]
        rew_explore = 2.0
        rew_mm = 1.0
        next_obs = {"state": 2}
        done = False

        self.buffer.store(obs, act_explore, act_mm, rew_explore, rew_mm, next_obs, done)

        self.assertEqual(self.buffer.size, 1)
        self.assertEqual(self.buffer.obs_buf[0], obs)
        self.assertEqual(self.buffer.acts_explore_buf[0], act_explore)
        self.assertEqual(self.buffer.acts_mm_buf[0], act_mm)
        self.assertEqual(self.buffer.rews_explore_buf[0], rew_explore)
        self.assertEqual(self.buffer.rews_mm_buf[0], rew_mm)
        self.assertEqual(self.buffer.next_obs_buf[0], next_obs)
        self.assertEqual(self.buffer.done_buf[0], done)

    def test_sample_batch(self):
        for i in range(50):
            self.buffer.store(
                {"state": i},
                i,
                [i],
                float(i) * 2,
                float(i),
                {"state": i + 1},
                i % 2 == 0,
            )

        batch = self.buffer.sample_batch()

        self.assertEqual(len(batch["obs"]), 32)
        self.assertEqual(len(batch["acts_explore"]), 32)
        self.assertEqual(len(batch["acts_mm"]), 32)
        self.assertEqual(len(batch["rews_explore"]), 32)
        self.assertEqual(len(batch["rews_mm"]), 32)
        self.assertEqual(len(batch["next_obs"]), 32)
        self.assertEqual(len(batch["done"]), 32)


class TestFindNonMaskedRows(unittest.TestCase):

    def test_simple_case(self):
        mask = torch.tensor([[[1, 1], [1, 1]], [[0, 0], [1, 1]], [[1, 1], [0, 0]]])
        expected = [torch.tensor([0, 1]), torch.tensor([1]), torch.tensor([0])]
        result = find_non_masked_rows(mask)
        self.assertEqual(len(result), len(expected))
        for res, exp in zip(result, expected):
            self.assertTrue(torch.equal(res, exp))

    def test_all_masked(self):
        mask = torch.zeros((2, 3, 4))
        expected = [torch.tensor([]), torch.tensor([])]
        result = find_non_masked_rows(mask)
        self.assertEqual(len(result), len(expected))
        for res, exp in zip(result, expected):
            self.assertTrue(torch.equal(res, exp))

    def test_all_unmasked(self):
        mask = torch.ones((2, 3, 4))
        expected = [torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])]
        result = find_non_masked_rows(mask)
        self.assertEqual(len(result), len(expected))
        for res, exp in zip(result, expected):
            self.assertTrue(torch.equal(res, exp))

    def test_mixed_case(self):
        mask = torch.tensor(
            [[[1, 0, 1], [0, 1, 0], [1, 1, 1]], [[0, 0, 0], [1, 1, 1], [0, 1, 0]]]
        )
        expected = [torch.tensor([0, 1, 2]), torch.tensor([1, 2])]
        result = find_non_masked_rows(mask)
        self.assertEqual(len(result), len(expected))
        for i, (res, exp) in enumerate(zip(result, expected)):
            print(f"Batch {i}:")
            print(f"Expected: {exp}")
            print(f"Result: {res}")
            self.assertTrue(torch.equal(res, exp))

    def test_single_batch(self):
        mask = torch.tensor([[[1, 1], [0, 0], [1, 0]]])
        expected = [torch.tensor([0, 2])]
        result = find_non_masked_rows(mask)
        self.assertEqual(len(result), len(expected))
        for res, exp in zip(result, expected):
            self.assertTrue(torch.equal(res, exp))

    def test_large_input(self):
        mask = torch.randint(0, 2, (10, 100, 50))
        result = find_non_masked_rows(mask)
        self.assertEqual(len(result), 10)
        for batch_result in result:
            self.assertTrue(torch.is_tensor(batch_result))
            self.assertTrue(batch_result.dim() == 1)
            self.assertTrue(batch_result.dtype == torch.long)
