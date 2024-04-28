"""Utility functions for DQN."""

import logging
import operator
import os
import random
from collections import deque
from typing import Callable, Deque, Literal

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

    """

    def __init__(
        self,
        observation_type: Literal["dict", "tensor"],
        size: int,
        obs_dim: tuple = None,
        batch_size: int = 32,
    ):
        """Initialize replay buffer.

        Args:
            observation_type: "dict" or "tensor"
            size: size of the buffer
            batch_size: batch size to sample

        """
        if batch_size > size:
            raise ValueError("batch_size must be smaller than size")
        if observation_type == "dict":
            self.obs_buf = np.array([{}] * size)
            self.next_obs_buf = np.array([{}] * size)
        else:
            raise ValueError("At the moment, observation_type must be 'dict'")
            # self.obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
            # self.next_obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)

        self.acts_buf = np.zeros([size], dtype=np.float32)
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
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


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
    samples: dict[str, dict[str, np.ndarray]],
    device: str,
    dqn: dict[str, torch.nn.Module],
    dqn_target: dict[str, torch.nn.Module],
    ddqn: str,
    gamma: float,
) -> dict[str, torch.Tensor]:
    """Return td loss.

    # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
    #       = r                       otherwise

    Args:
        samples: A dictionary of samples from the replay buffer.
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
        loss["mm"]: temporal difference loss value for mm
        loss["explore"]: temporal difference loss value for explore

    """
    loss = {"mm": None, "explore": None}

    for policy in ["mm", "explore"]:

        state = samples[policy]["obs"]
        next_state = samples[policy]["next_obs"]
        action = torch.LongTensor(samples[policy]["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples[policy]["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples[policy]["done"].reshape(-1, 1)).to(device)

        if policy == "mm":
            entity_of_interest = None
            edge_of_interest = {"relation_type": "any", "qualifier": "current_time"}
        elif policy == "explore":
            entity_of_interest = {"entity_type": "agent", "qualifier": "any"}
            edge_of_interest = None
        of_interest = {
            "entity_of_interest": entity_of_interest,
            "edge_of_interest": edge_of_interest,
        }

        gnn_features = dqn["gnn"](state, of_interest=of_interest)
        gnn_features = gnn_features.view(-1, gnn_features.size(-1))
        curr_q_value = dqn[policy](gnn_features).gather(1, action)

        if ddqn:
            next_q_value = (
                dqn_target[policy](dqn_target["gnn"](next_state, policy=policy))
                .gather(
                    1,
                    dqn[policy](dqn["gnn"](next_state, policy=policy)).argmax(
                        dim=1, keepdim=True
                    ),
                )
                .detach()
            )

        else:
            next_q_value = (
                dqn_target[policy](dqn_target["gnn"](next_state))
                .max(dim=1, keepdim=True)[0]
                .detach()
            )

        if policy == "mm":
            # This is done considering MARL.
            # Generate random permutation indices
            # Random permutation of indices along dimension 0 (rows)
            perm_indices = torch.randperm(next_q_value.size(0))

            # Permute rows of the tensor
            next_q_value = torch.index_select(next_q_value, 0, perm_indices)

            # Adjust the dimension of next_q_value to match curr_q_value
            min_size = min(curr_q_value.size(0), next_q_value.size(0))
            next_q_value = next_q_value[:min_size]
            curr_q_value = curr_q_value[:min_size]

        mask = 1 - done[:min_size]
        target = (reward[:min_size] + gamma * next_q_value * mask).to(device)

        # calculate dqn loss
        loss[policy] = F.smooth_l1_loss(curr_q_value, target)

    return loss

    #     mask = 1 - done
    #     target = (reward + gamma * next_q_value * mask).to(device)

    #     # calculate dqn loss
    #     loss[policy] = F.smooth_l1_loss(curr_q_value, target)

    # return loss


def update_model(
    replay_buffer: ReplayBuffer,
    optimizer: torch.optim.Adam,
    device: str,
    dqn: dict[str, torch.nn.Module],
    dqn_target: dict[str, torch.nn.Module],
    ddqn: str,
    gamma: float,
) -> dict[str, float]:
    """Update the model by gradient descent.

    Args:
        replay_buffer: replay buffer
        optimizer: optimizer
        device: cpu or cuda
        dqn: dqn model
        dqn_target: dqn target model
        ddqn: whether to use double dqn or not
        gamma: discount factor

    Returns:
        loss: temporal difference loss value
    """
    samples = {}
    for policy in ["mm", "explore"]:
        samples[policy] = replay_buffer[policy].sample_batch()

    loss = compute_loss(samples, device, dqn, dqn_target, ddqn, gamma)

    optimizer.zero_grad()
    total_loss = loss["mm"] + loss["explore"]
    total_loss.backward()
    optimizer.step()

    return {"mm": loss["mm"].item(), "explore": loss["explore"].item()}


def select_action(
    state: dict,
    greedy: bool,
    dqn: torch.nn.Module,
    epsilon: float,
    action_space: gym.spaces.Discrete,
) -> tuple[int, list]:
    """Select an action from the input state, with epsilon-greedy policy.

    Args:
        state: The current state of the memory systems. This is NOT what the gym env
        gives you. This is made by the agent.
        greedy: always pick greedy action if True
        save_q_value: whether to save the q values or not.

    Returns:
        selected_action: an action to take.
        q_values: a list of q values for each action.

    """
    q_values = dqn(np.array([state])).detach().cpu().tolist()[0]

    if greedy or epsilon < np.random.random():
        selected_action = argmax(q_values)
    else:
        selected_action = action_space.sample().item()

    return selected_action, q_values


def save_validation(
    scores_temp: list,
    scores: dict,
    default_root_dir: str,
    num_validation: int,
    val_filenames: dict,
    dqn=dict,
) -> None:
    """Keep the best validation model.

    Args:
        scores_temp: a list of validation scores for the current validation episode.
        scores: a dictionary of scores for train, validation, and test.
        default_root_dir: the root directory where the results are saved.
        num_validation: the current validation episode.
        val_filenames: best and current
        dqn: gnn, mm, and explore
    """
    current_score = round(np.mean(scores_temp).item())
    val_filenames["current"]["score"] = current_score
    best_score = val_filenames["best"]["score"]
    scores["val"].append(scores_temp)

    filename_prefix = os.path.join(
        default_root_dir, f"episode={num_validation}_val-score={current_score}"
    )
    for nn_type in ["gnn", "mm", "explore"]:
        filename = filename_prefix + "_" + nn_type + ".pt"
        val_filenames["current"][nn_type] = filename
        torch.save(dqn[nn_type].state_dict(), filename)

    if current_score > best_score:
        val_filenames["best"] = val_filenames["current"]

    scores_to_compare = []
    for filename in val_filenames:
        score = int(filename.split("val-score=")[-1].split("/")[-1])
        scores_to_compare.append(score)

    indexes = list_duplicates_of(scores_to_compare, max(scores_to_compare))
    if if_duplicate_take_first:
        file_to_keep = val_filenames[indexes[0]]
    else:
        file_to_keep = val_filenames[indexes[-1]]

    for filename in val_filenames:
        if filename != file_to_keep:
            os.remove(filename)
            val_filenames.remove(filename)


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
