"""Utility functions for DQN."""

import logging
import os
from typing import Literal
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from humemai.utils import (
    argmax,
    is_running_notebook,
    list_duplicates_of,
    write_pickle,
    write_yaml,
)
from IPython.display import clear_output
from tqdm.auto import tqdm

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ReplayBuffer:
    """A simple numpy replay buffer.

    numpy replay buffer is faster than deque or list.
    copied from https://github.com/Curt-Park/rainbow-is-all-you-need

    Attributes:
        obs_buf (np.ndarray): This is a np.ndarray of dictionaries. This is because
            dictionary can take any type of data.
        next_obs_buf (np.ndarray): This is a np.ndarray of dictionaries. This is because
            dictionary can take any type of data.
        acts_buf (np.ndarray): This is a np.ndarray of dictionaries. This is because
            dictionary can take any type of data. In our scenario, an action is a vector
            of integer whose length is variable.
        rews_buf (np.ndarray): This is a np.ndarray of floats.
        done_buf (np.ndarray): This is a np.ndarray of bools.
        max_size (int): The maximum size of the buffer.
        batch_size (int): The batch size to sample.
        ptr(int): The pointer to the current index.
        size (int): The current size of the buffer.

    Example:
    ```
    from agent.dqn.utils import ReplayBuffer
    import random

    buffer = ReplayBuffer(8, 4)

    for _ in range(6):

        rand_dict_1 = {str(i): str(random.randint(0, 10)) for i in range(3)}
        rand_dict_2 = {str(i): str(random.randint(0, 10)) for i in range(3)}
        action = [i for i in range(random.randint(1, 10))]
        buffer.store(
            *[
                rand_dict_1,
                action,
                random.choice([-1, 1]),
                rand_dict_2,
                random.choice([False, True]),
            ]
        )

    sample = buffer.sample_batch()
    ```
    >>> sample
    {'obs': array([{'0': '3', '1': '4', '2': '5'}, {'0': '5', '1': '1', '2': '0'},
            {'0': '9', '1': '6', '2': '2'}, {'0': '7', '1': '2', '2': '4'}],
        dtype=object),
    'next_obs': array([{'0': '5', '1': '0', '2': '5'}, {'0': '5', '1': '1', '2': '9'},
            {'0': '5', '1': '8', '2': '4'}, {'0': '6', '1': '9', '2': '4'}],
        dtype=object),
    'acts': array([list([0, 1, 2, 3, 4, 5]), list([0, 1, 2, 3, 4, 5, 6, 7]),
            list([0, 1]), list([0])], dtype=object),
    'rews': array([ 1., -1., -1.,  1.], dtype=float32),
    'done': array([0., 1., 1., 0.], dtype=float32)}

    >>> sample["acts"]
    array([list([0, 1, 2, 3, 4, 5, 6]), list([0, 1, 2]),
        list([0, 1, 2, 3, 4, 5, 6, 7]), list([0, 1, 2, 3])], dtype=object)

    >>> sample["acts"].shape
    (4,)

    """

    def __init__(
        self,
        size: int,
        batch_size: int = 32,
    ):
        """Initialize replay buffer.

        Args:
            size: size of the buffer
            batch_size: batch size to sample

        """
        if batch_size > size:
            raise ValueError("batch_size must be smaller than size")

        self.obs_buf = np.array([{}] * size)
        self.next_obs_buf = np.array([{}] * size)
        self.acts_buf = np.array([{}] * size)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        (
            self.ptr,
            self.size,
        ) = (
            0,
            0,
        )

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store the data in the buffer.

        Args:
            obs: observation
            act: action
            rew: reward
            next_obs: next observation
            done: done

        """
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_indices(self) -> np.ndarray:
        """Sample indices from the buffer."""
        return np.random.choice(self.size, size=self.batch_size, replace=False)

    def sample_batch(self, idxs: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Sample a batch of data from the buffer.

        Args:
            idxs: indices to sample from the buffer. If None, idxs will be
                randomly sampled.

        Returns:
            A dictionary of samples from the replay buffer.
                obs: np.ndarray,
                next_obs: np.ndarray,
                acts: np.ndarray,
                rews: float,
                done: bool

        """
        if idxs is None:
            idxs = self.sample_indices()
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


class MultiAgentReplayBuffer:
    """A simple numpy replay buffer for multi-agent."""

    def __init__(self, replay_buffer_1: ReplayBuffer, replay_buffer_2: ReplayBuffer):
        """Initialize replay buffer.

        Attributes:
            replay_buffer_1 (ReplayBuffer): replay buffer for memory management
            replay_buffer_2 (ReplayBuffer): replay buffer for explore

        Args:
            replay_buffer_1: replay buffer for memory management
            replay_buffer_2: replay buffer for explore

        """
        self.replay_buffer_1 = replay_buffer_1
        self.replay_buffer_2 = replay_buffer_2
        assert len(replay_buffer_1) == len(
            replay_buffer_2
        ), "The replay buffers must have the same size."
        assert (
            replay_buffer_1.batch_size == replay_buffer_2.batch_size
        ), "The replay buffers must have the same batch size."

    def sample_batch(self, sample_same_index: bool) -> tuple[dict, dict]:
        """Sample a batch of data from the buffer.

        Ars:
            sample_same_index: whether to sample the same index for both replay buffers.

        Returns:
            batch_1: a dictionary of samples from the replay buffer 1.
                obs: np.ndarray,
                next_obs: np.ndarray,
                acts: np.ndarray,
                rews: float,
                done: bool

            batch_2: a dictionary of samples from the replay buffer 2.
                obs: np.ndarray,
                next_obs: np.ndarray,
                acts: np.ndarray,
                rews: float,
                done: bool

        """

        if sample_same_index:
            idxs = self.replay_buffer_1.sample_indices()
            batch_1 = self.replay_buffer_1.sample_batch(idxs)
            batch_2 = self.replay_buffer_2.sample_batch(idxs)
        else:
            batch_1 = self.replay_buffer_1.sample_batch()
            batch_2 = self.replay_buffer_2.sample_batch()

        return batch_1, batch_2


def plot_results(
    scores: dict,
    training_loss: list,
    epsilons: list,
    q_values: dict,
    iteration_idx: int,
    number_of_actions: int,
    num_iterations: int,
    total_maximum_episode_rewards: int,
    default_root_dir: str,
    to_plot: str = "all",
    save_fig: bool = False,
) -> None:
    """Plot things for DQN training.

    Args:
        to_plot: what to plot:
            training_td_loss
            epsilons
            scores
            q_values_train
            q_values_val
            q_values_test

    """
    is_notebook = is_running_notebook()

    if is_notebook:
        clear_output(True)

    if to_plot == "all":
        plt.figure(figsize=(20, 20))

        plt.subplot(333)
        if scores["train"]:
            plt.title(
                f"iteration {iteration_idx} out of {num_iterations}. "
                f"training score: {scores['train'][-1]} out of "
                f"{total_maximum_episode_rewards}"
            )
            plt.plot(scores["train"], label="Training score")
            plt.xlabel("episode")

        if scores["val"]:
            val_means = [round(np.mean(scores).item()) for scores in scores["val"]]
            plt.title(
                f"validation score: {val_means[-1]} out of "
                f"{total_maximum_episode_rewards}"
            )
            plt.plot(val_means, label="Validation score")
            plt.xlabel("episode")

        if scores["test"]:
            plt.title(
                f"test score: {np.mean(scores['test'])} out of "
                f"{total_maximum_episode_rewards}"
            )
            plt.plot(
                [round(np.mean(scores["test"]).item(), 2)] * len(scores["train"]),
                label="Test score",
            )
            plt.xlabel("episode")
        plt.legend(loc="upper left")

        plt.subplot(331)
        plt.title("training td loss")
        plt.plot(training_loss)
        plt.xlabel("update counts")

        plt.subplot(332)
        plt.title("epsilons")
        plt.plot(epsilons)
        plt.xlabel("update counts")

        for subplot_num, split in zip([334, 335, 336], ["train", "val", "test"]):
            plt.subplot(subplot_num)
            plt.title(f"Q-values (mm), {split}")
            for action_number in range(number_of_actions):
                plt.plot(
                    [q_values_[action_number] for q_values_ in q_values[split]["mm"]],
                    label=f"action {action_number}",
                )
            plt.legend(loc="upper left")
            plt.xlabel("number of actions")

        for subplot_num, split in zip([337, 338, 339], ["train", "val", "test"]):
            plt.subplot(subplot_num)
            plt.title(f"Q-values (explore), {split}")
            for action_number in range(number_of_actions):
                plt.plot(
                    [
                        q_values_[action_number]
                        for q_values_ in q_values[split]["explore"]
                    ],
                    label=f"action {action_number}",
                )
            plt.legend(loc="upper left")
            plt.xlabel("number of actions")

        plt.subplots_adjust(hspace=0.5)
        if save_fig:
            plt.savefig(os.path.join(default_root_dir, "plot.pdf"))

        if is_notebook:
            plt.show()
        else:
            console(**locals())
            plt.close("all")

    elif to_plot == "training_td_loss":
        plt.figure()
        plt.title("training td loss")
        plt.plot(training_loss)
        plt.xlabel("update counts")

    elif to_plot == "epsilons":
        plt.figure()
        plt.title("epsilons")
        plt.plot(epsilons)
        plt.xlabel("update counts")

    elif to_plot == "scores":
        plt.figure()

        if scores["train"]:
            plt.title(
                f"iteration {iteration_idx} out of {num_iterations}. "
                f"training score: {scores['train'][-1]} out of "
                f"{total_maximum_episode_rewards}"
            )
            plt.plot(scores["train"], label="Training score")
            plt.xlabel("episode")

        if scores["val"]:
            val_means = [round(np.mean(scores).item()) for scores in scores["val"]]
            plt.title(
                f"validation score: {val_means[-1]} out of "
                f"{total_maximum_episode_rewards}"
            )
            plt.plot(val_means, label="Validation score")
            plt.xlabel("episode")

        if scores["test"]:
            plt.title(
                f"test score: {np.mean(scores['test'])} out of "
                f"{total_maximum_episode_rewards}"
            )
            plt.plot(
                [round(np.mean(scores["test"]).item(), 2)] * len(scores["train"]),
                label="Test score",
            )
            plt.xlabel("episode")
        plt.legend(loc="upper left")

    else:
        pass

    for split in ["train", "val", "test"]:
        if to_plot == f"q_values_{split}":
            plt.figure(figsize=(20, 13))
            for subplot_num, policy in zip([121, 122], ["mm", "explore"]):
                plt.subplot(subplot_num)
                plt.title(f"Q-values ({policy}), {split}")
                for action_number in range(number_of_actions):
                    plt.plot(
                        [
                            q_values_[action_number]
                            for q_values_ in q_values[split][policy]
                        ],
                        label=f"action {action_number}",
                    )
                plt.legend(loc="upper left")
                plt.xlabel("number of actions")


def console(
    scores: dict,
    training_loss: list,
    iteration_idx: int,
    num_iterations: int,
    total_maximum_episode_rewards: int,
    **kwargs,
) -> None:
    """Print the dqn training to the console."""
    if scores["train"]:
        tqdm.write(
            f"iteration {iteration_idx} out of {num_iterations}.\n"
            f"training score: "
            f"{scores['train'][-1]} out of {total_maximum_episode_rewards}"
        )

    if scores["val"]:
        val_means = [round(np.mean(scores).item()) for scores in scores["val"]]
        tqdm.write(
            f"validation score: {val_means[-1]} "
            f"out of {total_maximum_episode_rewards}"
        )

    if scores["test"]:
        tqdm.write(
            f"test score: {np.mean(scores['test'])} out of "
            f"{total_maximum_episode_rewards}"
        )

    tqdm.write(f"training loss: {training_loss[-1]}\n")
    print()


def save_final_results(
    scores: dict,
    training_loss: list,
    default_root_dir: str,
    q_values: dict,
    self: object,
) -> None:
    """Save dqn train / val / test results."""
    results = {
        "train_score": scores["train"],
        "validation_score": [
            {
                "mean": round(np.mean(scores).item(), 2),
                "std": round(np.std(scores).item(), 2),
            }
            for scores in scores["val"]
        ],
        "test_score": {
            "mean": round(np.mean(scores["test"]).item(), 2),
            "std": round(np.std(scores["test"]).item(), 2),
        },
        "training_loss": training_loss,
    }
    write_yaml(results, os.path.join(default_root_dir, "results.yaml"))
    write_yaml(q_values, os.path.join(default_root_dir, "q_values.yaml"))
    write_pickle(self, os.path.join(default_root_dir, "agent.pkl"))


def compute_loss(
    batch_mm: dict,
    batch_explore: dict,
    device: str,
    dqn: dict[str, torch.nn.Module],
    dqn_target: dict[str, torch.nn.Module],
    ddqn: str,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the DQN td loss.

    G_t   = r + gamma * v(s_{t+1})  if state != Terminal
          = r                       otherwise

    Args:
        batch: A dictionary of samples from the replay buffer.
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
        device: cpu or cuda
        dqn: dqn model
        dqn_target: dqn target model
        ddqn: whether to use double dqn or not
        gamma: discount factor

    Returns:
        loss_mm, loss_explore: TD losses for memory management and explore

    """
    s_mm = batch_mm["obs"]
    s_next_mm = batch_mm["next_obs"]
    a_mm = batch_mm["acts"]
    r_mm = torch.FloatTensor(batch_mm["rews"].reshape(-1, 1)).to(device)
    d_mm = torch.FloatTensor(batch_mm["done"].reshape(-1, 1)).to(device)

    s_explore = batch_explore["obs"]
    s_next_explore = batch_explore["next_obs"]
    a_explore = batch_explore["acts"]
    r_explore = torch.FloatTensor(batch_explore["rews"].reshape(-1, 1)).to(device)
    d_explore = torch.FloatTensor(batch_explore["done"].reshape(-1, 1)).to(device)

    assert np.array_equal(r_mm, r_explore), "Rewards are not the same."
    assert np.array_equal(d_mm, d_explore), "Dones are not the same."

    # MEMORY MANAGEMENT
    curr_q_value_mm = dqn(s_mm, policy="mm")  # [batch_size, num_actions]

    # EXPLORE
    curr_q_value_explore = dqn(s_mm, policy="explore")

    a_explore = (
        torch.LongTensor([item[0] for item in a_explore]).reshape(-1, 1).to(device)
    )
    curr_q_value_explore = curr_q_value_explore.gather(dim=1, index=a_explore)

    if ddqn:
        next_q_value_explore = (
            dqn_target(s_next_explore, policy="explore")
            .gather(
                1, dqn(s_next_explore, policy="explore").argmax(dim=1, keepdim=True)
            )
            .detach()
        )
    else:
        next_q_value_explore = (
            dqn_target(s_next_explore, policy="explore")
            .max(dim=1, keepdim=True)[0]
            .detach()
        )
    mask_explore = 1 - d_explore
    target = (r_explore + gamma * next_q_value_explore * mask_explore).to(device)

    loss_explore = F.smooth_l1_loss(curr_q_value_explore, target)

    return loss_mm, loss_explore


def update_model(
    replay_buffer_mm: ReplayBuffer,
    replay_buffer_explore: ReplayBuffer,
    optimizer: torch.optim.Adam,
    device: str,
    dqn: dict[str, torch.nn.Module],
    dqn_target: dict[str, torch.nn.Module],
    ddqn: str,
    gamma: float,
    loss_weights: tuple[float, float] = [0.5, 0.5],
) -> tuple[float, float, float]:
    """Update the model by gradient descent.

    Args:
        replay_buffer_mm: replay buffer for memory management
        replay_buffer_explore: replay buffer for explore
        optimizer: optimizer
        device: cpu or cuda
        dqn: dqn model
        dqn_target: dqn target model
        ddqn: whether to use double dqn or not
        gamma: discount factor
        loss_weights: weights for mm and explore losses

    Returns:
        loss_mm, loss_explore, loss_combined: TD losses for memory management,
            explore and combined

    """
    buffer_combined = MultiAgentReplayBuffer(replay_buffer_mm, replay_buffer_explore)
    batch_mm, batch_explore = buffer_combined.sample_batch(sample_same_index=True)

    loss_mm, loss_explore = compute_loss(
        batch_mm, batch_explore, device, dqn, dqn_target, ddqn, gamma
    )

    loss = loss_weights[0] * loss_mm + loss_weights[1] * loss_explore

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_mm = loss_mm.detach().cpu().numpy().item()
    loss_explore = loss_explore.detach().cpu().numpy().item()
    loss = loss.detach().cpu().numpy().item()

    return loss_mm, loss_explore, loss


def select_action(
    state: dict,
    greedy: bool,
    dqn: torch.nn.Module,
    epsilon: float,
    policy_type: Literal["mm", "explore", "qa"],
) -> tuple[list[int], list[list[float]]]:
    """Select action(s) from the input state, with epsilon-greedy policy.

    Args:
        state: The current state of the memory systems. This is NOT what the gym env
            gives you. This is made by the agent. This shouldn't be a batch of samples,
            but a single sample without a batch dimension.
        greedy: always pick greedy action if True
        dqn: dqn model
        epsilon: epsilon
        policy_type: mm, explore, or qa

    Returns:
        selected_actions: dimension is [num_actions_taken]
        q_values: dimension is [num_actions_taken, action_space_dim]

    """
    # Since dqn requires a batch dimension, we need to encapsulate the state in a list
    q_values = dqn(np.array([state]), policy_type=policy_type).detach().cpu().tolist()

    # remove the dummy batch dimension to make [num_actions_taken, action_space_dim]
    q_values = q_values[0]

    action_space_dim = len(q_values[0])

    if greedy or epsilon < np.random.random():
        selected_actions = [argmax(q_values_) for q_values_ in q_values]
    else:
        selected_actions = [random.randint(0, action_space_dim - 1) for _ in q_values]

    return selected_actions, q_values


def save_validation(
    scores_temp: list,
    scores: dict,
    default_root_dir: str,
    num_validation: int,
    val_filenames: dict[str, str],
    dqn=torch.nn.Module,
) -> None:
    """Keep the best validation model.

    Args:
        scores_temp: a list of validation scores for the current validation episode.
        scores: a dictionary of scores for train, validation, and test.
        default_root_dir: the root directory where the results are saved.
        num_validation: the current validation episode.
        val_filenames: looks like `{"best": None, "last": None}`
        dqn: gnn, mm, and explore

    """
    scores["val"].append(scores_temp)
    last_score = round(np.mean(scores_temp).item())
    filename = os.path.join(
        default_root_dir, f"episode={num_validation}_val-score={last_score}.pt"
    )
    val_filenames["last"] = filename

    if val_filenames["best"] is None:
        val_filenames["best"] = filename
        torch.save(dqn.state_dict(), filename)

    else:
        best_score = int(val_filenames["best"].split("val-score=")[-1].split(".pt")[0])

        if last_score > best_score:
            os.remove(val_filenames["best"])
            val_filenames["best"] = filename
            torch.save(dqn.state_dict(), filename)


def save_states_q_values_actions(
    states: list,
    q_values: list,
    actions: list,
    default_root_dir: str,
    val_or_test: str,
    num_validation: int | None = None,
) -> None:
    """Save states, q_values, and actions.

    Args:
        states: a list of states.
        q_values: a list of q_values.
        actions: a list of actions.
        default_root_dir: the root directory where the results are saved.
        val_or_test: "val" or "test"
        num_validation: the current validation episode.

    """
    if val_or_test.lower() == "val":
        filename = os.path.join(
            default_root_dir,
            f"states_q_values_actions_val_episode={num_validation}.yaml",
        )
    else:
        filename = os.path.join(default_root_dir, "states_q_values_actions_test.yaml")

    assert len(states) == len(q_values) == len(actions)
    to_save = [
        {"state": s, "q_values": q, "action": a}
        for s, q, a in zip(states, q_values, actions)
    ]
    write_yaml(to_save, filename)


def target_hard_update(dqn: torch.nn.Module, dqn_target: torch.nn.Module) -> None:
    """Hard update: target <- local.

    Args:
        dqn: dqn model
        dqn_target: dqn target model
    """
    dqn_target.load_state_dict(dqn.state_dict())


def update_epsilon(
    epsilon: float, max_epsilon: float, min_epsilon: float, epsilon_decay_until: int
) -> float:
    """Linearly decrease epsilon

    Args:
        epsilon: current epsilon
        max_epsilon: initial epsilon
        min_epsilon: minimum epsilon
        epsilon_decay_until: the last iteration index to decay epsilon

    Returns:
        epsilon: updated epsilon

    """
    epsilon = max(
        min_epsilon, epsilon - (max_epsilon - min_epsilon) / epsilon_decay_until
    )

    return epsilon
