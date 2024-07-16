import unittest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from agent.dqn.utils import (
    ReplayBuffer,
    MultiAgentReplayBuffer,
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
        act = [0, 1]
        rew = 1.0
        next_obs = {"state": 2}
        done = False

        self.buffer.store(obs, act, rew, next_obs, done)

        self.assertEqual(self.buffer.size, 1)
        self.assertEqual(self.buffer.obs_buf[0], obs)
        self.assertEqual(self.buffer.acts_buf[0], act)
        self.assertEqual(self.buffer.rews_buf[0], rew)
        self.assertEqual(self.buffer.next_obs_buf[0], next_obs)
        self.assertEqual(self.buffer.done_buf[0], done)

    def test_sample_batch(self):
        for i in range(50):
            self.buffer.store({"state": i}, [i], float(i), {"state": i + 1}, i % 2 == 0)

        batch = self.buffer.sample_batch()

        self.assertEqual(len(batch["obs"]), 32)
        self.assertEqual(len(batch["acts"]), 32)
        self.assertEqual(len(batch["rews"]), 32)
        self.assertEqual(len(batch["next_obs"]), 32)
        self.assertEqual(len(batch["done"]), 32)


class TestMultiAgentReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer1 = ReplayBuffer(size=100, batch_size=32)
        self.buffer2 = ReplayBuffer(size=100, batch_size=32)
        self.multi_buffer = MultiAgentReplayBuffer(self.buffer1, self.buffer2)

    def test_sample_batch_same_index(self):
        for i in range(50):
            self.buffer1.store(
                {"state": i}, [i], float(i), {"state": i + 1}, i % 2 == 0
            )
            self.buffer2.store(
                {"state": i * 2},
                [i * 2],
                float(i * 2),
                {"state": (i + 1) * 2},
                i % 3 == 0,
            )

        batch1, batch2 = self.multi_buffer.sample_batch(sample_same_index=True)

        self.assertEqual(len(batch1["obs"]), 32)
        self.assertEqual(len(batch2["obs"]), 32)
        self.assertEqual(batch1["rews"].shape, batch2["rews"].shape)

    def test_sample_batch_different_index(self):
        for i in range(50):
            self.buffer1.store(
                {"state": i}, [i], float(i), {"state": i + 1}, i % 2 == 0
            )
            self.buffer2.store(
                {"state": i * 2},
                [i * 2],
                float(i * 2),
                {"state": (i + 1) * 2},
                i % 3 == 0,
            )

        batch1, batch2 = self.multi_buffer.sample_batch(sample_same_index=False)

        self.assertEqual(len(batch1["obs"]), 32)
        self.assertEqual(len(batch2["obs"]), 32)
        # Rewards should likely be different when sampling from different indices
        self.assertFalse(np.array_equal(batch1["rews"], batch2["rews"]))

    def test_sample_batch_with_specific_indices(self):
        for i in range(50):
            self.buffer1.store(
                {"state": i}, [i], float(i), {"state": i + 1}, i % 2 == 0
            )
            self.buffer2.store(
                {"state": i * 2},
                [i * 2],
                float(i * 2),
                {"state": (i + 1) * 2},
                i % 3 == 0,
            )

        # Create specific indices to sample
        specific_indices = np.random.choice(50, size=32, replace=False)

        batch1 = self.buffer1.sample_batch(idxs=specific_indices)
        batch2 = self.buffer2.sample_batch(idxs=specific_indices)

        self.assertEqual(len(batch1["obs"]), 32)
        self.assertEqual(len(batch2["obs"]), 32)

        # Check if the sampled observations match the expected ones
        for i, idx in enumerate(specific_indices):
            self.assertEqual(batch1["obs"][i], {"state": idx})
            self.assertEqual(batch2["obs"][i], {"state": idx * 2})


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

