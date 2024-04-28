"""Agent that uses a GNN for its DQN, for the RoomEnv2 environment."""

import datetime
import os
from copy import deepcopy
import shutil

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from humemai.utils import is_running_notebook, write_yaml
from humemai.policy import encode_all_observations, manage_memory
from humemai.memory import EpisodicMemory, MemorySystems, SemanticMemory, ShortMemory
from humemai.policy import answer_question, encode_observation, explore, manage_memory
from room_env.envs.room2 import RoomEnv2

from .nn import GNN
from .nn import MLP
from .utils import (
    ReplayBuffer,
    plot_results,
    save_final_results,
    save_states_q_values_actions,
    save_validation,
    select_action,
)


class DQNAgent:
    """DQN Agent interacting with environment.

    This is an upgrade from https://github.com/humemai/agent-room-env-v2-lstm. The two
    policies, i.e., memory management and exploration, are learned by a GNN at once!
    The question-policy is still hand-crafted. Potentially, this can be learned by a
    contextual bandit algorithm.

    Based on https://github.com/Curt-Park/rainbow-is-all-you-need/
    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v2",
        num_iterations: int = 10000,
        replay_buffer_size: int = 10000,
        warm_start: int = 1000,
        batch_size: int = 32,
        target_update_interval: int = 10,
        epsilon_decay_until: float = 10000,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.9,
        capacity: dict = {
            "episodic": 16,
            "semantic": 16,
            "short": 1,
        },
        agent_to_episodic: bool = True,
        pretrain_semantic: str | bool = False,
        gnn_params: dict = {},
        mlp_params: dict = {},
        run_test: bool = True,
        num_samples_for_results: int = 10,
        plotting_interval: int = 10,
        train_seed: int = 5,
        test_seed: int = 0,
        device: str = "cpu",
        mm_policy: str = "generalize",
        qa_policy: str = "episodic_semantic",
        explore_policy: str = "avoid_walls",
        env_config: dict = {
            "question_prob": 1.0,
            "terminates_at": 99,
            "randomize_observations": "objects",
            "room_size": "l",
            "rewards": {"correct": 1, "wrong": 0, "partial": 0},
            "make_everything_static": False,
            "num_total_questions": 1000,
            "question_interval": 1,
            "include_walls_in_observations": True,
        },
        ddqn: bool = True,
        default_root_dir: str = "./training-results/stochastic-objects/DQN/",
        run_handcrafted_baselines: bool = False,
    ) -> None:
        """Initialization.

        Args:
            env_str: environment string. This has to be "room_env:RoomEnv-v2"
            num_iterations: number of iterations to train
            replay_buffer_size: size of replay buffer
            warm_start: number of steps to fill the replay buffer, before training
            batch_size: This is the amount of samples sampled from the replay buffer.
            target_update_interval: interval to update target network
            epsilon_decay_until: until which iteration to decay epsilon
            max_epsilon: maximum epsilon
            min_epsilon: minimum epsilon
            gamma: discount factor
            capacity: The capacity of each human-like memory systems
            agent_to_episodic: If true, the agent's location related observations will
                be stored in its episodic memory system. If false, the agent decides
                what to do with its memory management policy
            pretrain_semantic: whether to pretrain the semantic memory system.
            gnn_params: parameters for the neural network (GNN)
            mlp_params: parameters for the neural network (MLP)
            run_test: whether to run test
            num_samples_for_results: The number of samples to validate / test the agent.
            plotting_interval: interval to plot results
            train_seed: seed for training
            test_seed: seed for testing
            device: This is either "cpu" or "cuda".
            mm_policy: memory management policy. Choose one of "generalize", "random",
                "rl", or "neural"
            qa_policy: question answering policy Choose one of "episodic_semantic",
                "random", or "neural". qa_policy shouldn't be trained with RL. There is
                no sequence of states / actions to learn from.
            explore_policy: The room exploration policy. Choose one of "random",
                "avoid_walls", "rl", or "neural"
            env_config: The configuration of the environment.
                question_prob: The probability of a question being asked at every
                    observation.
                terminates_at: The maximum number of steps to take in an episode.
                seed: seed for env
                room_size: The room configuration to use. Choose one of "dev", "xxs",
                    "xs", "s", "m", or "l".
            ddqn: whether to use double DQN
            default_root_dir: default root directory to save results
            run_handcrafted_baselines: whether to run handcrafted baselines

        """
        params_to_save = deepcopy(locals())
        del params_to_save["self"]
        self._create_directory(params_to_save)

        self.train_seed = train_seed
        self.test_seed = test_seed
        env_config["seed"] = self.train_seed

        self.env_str = env_str
        self.env_config = env_config
        self.mm_policy = mm_policy
        assert self.mm_policy in [
            "random",
            "episodic",
            "semantic",
            "generalize",
            "rl",
            "neural",
        ]
        self.qa_policy = qa_policy
        assert self.qa_policy in [
            "episodic_semantic",
            "episodic",
            "semantic",
            "random",
            "neural",
        ]
        self.explore_policy = explore_policy
        assert self.explore_policy in [
            "random",
            "avoid_walls",
            "new_room",
            "rl",
            "neural",
        ]
        self.num_samples_for_results = num_samples_for_results
        self.capacity = capacity
        self.agent_to_episodic = agent_to_episodic
        self.pretrain_semantic = pretrain_semantic
        self.env = gym.make(self.env_str, **self.env_config)
        self.default_root_dir = os.path.join(
            default_root_dir, str(datetime.datetime.now())
        )

        self.device = torch.device(device)
        print(f"Running on {self.device}")

        self.ddqn = ddqn
        self.val_filenames = {"gnn": None, "mm": None, "explore": None, "score": None}
        self.is_notebook = is_running_notebook()
        self.num_iterations = num_iterations
        self.plotting_interval = plotting_interval
        self.run_test = run_test

        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_until = epsilon_decay_until
        self.target_update_interval = target_update_interval
        self.gamma = gamma
        self.warm_start = warm_start
        assert self.batch_size <= self.warm_start <= self.replay_buffer_size

        self.gnn_params["device"] = self.device
        self.gnn_params["entities"] = self.env.unwrapped.entities
        self.gnn_params["relations"] = self.env.unwrapped.relations
        self.mlp_params["device"] = self.device
        self.dqn = {
            "gnn": GNN(**self.gnn_params),
            "mm": MLP(**self.mlp_params, num_outputs=3),
            "explore": MLP(**self.mlp_params, num_outputs=5),
        }
        self.dqn_target = {
            "gnn": GNN(**self.gnn_params),
            "mm": MLP(**self.mlp_params, num_outputs=3),
            "explore": MLP(**self.mlp_params, num_outputs=5),
        }
        for nn_type in ["gnn", "mm", "explore"]:
            self.dqn_target[nn_type].load_state_dict(self.dqn[nn_type].state_dict())
            self.dqn_target[nn_type].eval()

        self.replay_buffer = {}
        for policy_type in ["mm", "explore"]:
            self.replay_buffer[policy_type] = ReplayBuffer(
                observation_type="dict", size=replay_buffer_size, batch_size=batch_size
            )

        # optimizer
        self.optimizer = optim.Adam(
            list(self.dqn["gnn"].parameters())
            + list(self.dqn["mm"].parameters())
            + list(self.dqn["explore"].parameters())
        )

        self.q_values = {
            "train": {"mm": [], "explore": []},
            "val": {"mm": [], "explore": []},
            "test": {"mm": [], "explore": []},
        }

        if run_handcrafted_baselines:
            self.run_handcrafted_baselines()

    def _create_directory(self, params_to_save: dict) -> None:
        """Create the directory to store the results."""
        os.makedirs(self.default_root_dir, exist_ok=True)
        write_yaml(params_to_save, os.path.join(self.default_root_dir, "train.yaml"))

    def remove_results_from_disk(self) -> None:
        """Remove the results from the disk."""
        shutil.rmtree(self.default_root_dir)

    def init_memory_systems(self) -> None:
        """Initialize the agent's memory systems. This has nothing to do with the
        replay buffer."""
        self.memory_systems = MemorySystems(
            episodic=EpisodicMemory(
                capacity=self.capacity["episodic"], remove_duplicates=False
            ),
            semantic=SemanticMemory(capacity=self.capacity["semantic"]),
            short=ShortMemory(capacity=self.capacity["short"]),
        )

        assert self.pretrain_semantic in [False, "exclude_walls", "include_walls"]
        if self.pretrain_semantic in ["exclude_walls", "include_walls"]:
            if self.pretrain_semantic == "exclude_walls":
                exclude_walls = True
            else:
                exclude_walls = False
            room_layout = self.env.unwrapped.return_room_layout(exclude_walls)

            assert self.capacity["semantic"] > 0
            _ = self.memory_systems.semantic.pretrain_semantic(
                semantic_knowledge=room_layout,
                return_remaining_space=False,
                freeze=False,
            )

    def get_deepcopied_memory_state(self) -> dict:
        """Get a deepcopied memory state.

        This is necessary because the memory state is a list of dictionaries, which is
        mutable.

        Returns:
            deepcopied memory_state
        """
        return deepcopy(self.memory_systems.return_as_a_dict_list())

    def fill_replay_buffer(self) -> None:
        """Make the replay buffer full in the beginning with the uniformly-sampled
        actions. The filling continues until it reaches the warm start size.

        """
        raise NotImplementedError("Should be implemented by the inherited class!")

    def train(self) -> None:
        """Train the explore agent."""
        raise NotImplementedError("Should be implemented by the inherited class!")

    def validate(self) -> None:
        for nn_type in ["gnn", "mm", "explore"]:
            self.dqn[nn_type].eval()
        scores_temp, states, q_values, actions = self.validate_test_middle("val")

        save_validation(
            scores_temp=scores_temp,
            scores=self.scores,
            default_root_dir=self.default_root_dir,
            num_validation=self.num_validation,
            val_filenames=self.val_filenames,
            dqn=self.dqn,
        )
        save_states_q_values_actions(
            states, q_values, actions, self.default_root_dir, "val", self.num_validation
        )
        self.env.close()
        self.num_validation += 1
        for nn_type in ["gnn", "mm", "explore"]:
            self.dqn[nn_type].train()

    def test(self, checkpoint: dict[str, str] | None = None) -> None:
        """Test the agent.

        Args:
            checkpoint: use if None

        """
        for nn_type in ["gnn", "mm", "explore"]:
            if checkpoint is not None:
                self.dqn[nn_type].load_state_dict(torch.load(self.checkpoint[nn_type]))
            else:
                self.dqn[nn_type].load_state_dict(
                    torch.load(self.val_filenames["best"][nn_type])
                )
            self.dqn[nn_type].eval()

        self.env_config["seed"] = self.test_seed
        self.env = gym.make(self.env_str, **self.env_config)

        scores, states, q_values, actions = self.validate_test_middle("test")
        self.scores["test"] = scores

        save_final_results(
            self.scores, self.training_loss, self.default_root_dir, self.q_values, self
        )
        save_states_q_values_actions(
            states, q_values, actions, self.default_root_dir, "test"
        )

        self.plot_results("all", save_fig=True)
        self.env.close()
        for nn_type in ["gnn", "mm", "explore"]:
            self.dqn[nn_type].train()

    def plot_results(self, to_plot: str = "all", save_fig: bool = False) -> None:
        """Plot things for DQN training.

        Args:
            to_plot: what to plot:
                training_td_loss
                epsilons
                training_score
                validation_score
                test_score
                q_values_train
                q_values_val
                q_values_test

        """
        plot_results(
            self.scores,
            self.training_loss,
            self.epsilons,
            self.q_values,
            self.iteration_idx,
            self.action_space.n.item(),
            self.num_iterations,
            self.env.unwrapped.total_maximum_episode_rewards,
            self.default_root_dir,
            to_plot,
            save_fig,
        )
