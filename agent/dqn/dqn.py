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

    This is an upgrade from https://github.com/humemai/agent-room-env-v2-lstm.
    All three policies (memory management, question answering, and exploration) are
    learned by a GNN at once!

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
        nn_params: dict = {
            "hidden_size": 64,
            "num_layers": 2,
            "embedding_dim": 64,
            "make_categorical_embeddings": False,
            "memory_of_interest": [
                "episodic",
                "semantic",
                "short",
            ],
            "fuse_information": "sum",
            "include_positional_encoding": True,
            "max_timesteps": 100,
            "max_strength": 100,
        },
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
            nn_params: parameters for the neural network (DQN)
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

        self.nn_params = nn_params
        self.nn_params["capacity"] = self.capacity
        self.nn_params["device"] = self.device
        self.nn_params["entities"] = self.env.unwrapped.entities
        self.nn_params["relations"] = self.env.unwrapped.relations

        self.val_filenames = []
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

        self.dqn = GNN(**self.nn_params)
        self.dqn_target = GNN(**self.nn_params)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.replay_buffer = {}
        for policy_type in ["mm", "qa", "explore"]:
            self.replay_buffer[policy_type] = ReplayBuffer(
                observation_type="dict", size=replay_buffer_size, batch_size=batch_size
            )

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        self.q_values = {"train": [], "val": [], "test": []}

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

    def step(
        self, observations: dict, greedy: bool, save_to_replay_buffer: bool
    ) -> tuple[dict, int, float, bool, list]:
        """Run one step of the agent.

        Since there are three policies to learn, this is quite complicated.

        Args:
            observations: the observations from the environment
            greedy: whether to act greedily
            save_to_replay_buffer: whether to save the transition to the replay buffer

        Returns:
            observations, action, reward, done, q_values

        """
        states = {"mm": [], "qa": None, "explore": None}
        actions = {"mm": [], "qa": None, "explore": None}
        next_states = {"mm": [], "qa": None, "explore": None}
        q_values = {"mm": [], "qa": None, "explore": None}

        if save_to_replay_buffer:
            transitions = {}
            for policy_type in ["mm", "explore", "qa"]:
                transitions[policy_type] = []

        if self.agent_to_episodic:
            agent_location_observations = []
            remaining_room_observations = []

            for obs in observations["room"]:
                if obs[0] == "agent" and obs[1] == "atlocation":
                    agent_location_observations.append(obs)
                else:
                    remaining_room_observations.append(obs)

            if len(agent_location_observations) > 0:
                for obs in agent_location_observations:
                    encode_observation(self.memory_systems, obs)
                    manage_memory(
                        self.memory_systems,
                        "episodic",
                        split_possessive=False,
                    )
        else:
            remaining_room_observations = observations["room"]

        encode_all_observations(self.memory_systems, remaining_room_observations)

        # Memory management policy. There are multiple agents for this policy.
        num_mm_agents = len(remaining_room_observations)
        working_memory = self.get_working_memory()

        for idx in range(num_mm_agents):
            state = deepcopy(working_memory) + idx

            action, q_values_ = select_action(
                state=state,
                greedy=greedy,
                gnn=self.gnn,
                dqn=self.dqn["mm"],
                epsilon=self.epsilon,
                action_space=self.action_space["mm"],
            )
            states["mm"].append(state)
            actions["mm"].append(action)
            q_values["mm"].append(q_values_)

        for state, action in zip(states["mm"], actions["mm"]):
            manage_memory(
                self.memory_systems,
                self.action2str["mm"][action],
                split_possessive=False,
            )

        long_term_memory = self.get_long_term_memory()

        # Question answering policy. There is only one agent for this policy.
        state = deepcopy(long_term_memory) + observations["questions"]
        action, q_values_ = select_action(
            state=state,
            greedy=greedy,
            gnn=self.gnn,
            dqn=self.dqn["qa"],
            epsilon=self.epsilon,
            action_space=self.action_space["qa"],
        )
        states["qa"] = state
        actions["qa"] = action
        q_values["qa"] = q_values_

        # Exploration policy. There is only one agent for this policy.
        state = deepcopy(long_term_memory)
        action, q_values_ = select_action(
            state=state,
            greedy=greedy,
            gnn=self.gnn,
            dqn=self.dqn["explore"],
            epsilon=self.epsilon,
            action_space=self.action_space["explore"],
        )
        states["explore"] = state
        actions["explore"] = action
        q_values["explore"] = q_values_

        action_pair = (
            self.action2str["qa"][actions["qa"]],
            self.action2str["explore"][actions["explore"]],
        )

        (
            observations,
            reward,
            done,
            truncated,
            info,
        ) = self.env.step(action_pair)
        done = done or truncated

        # don't know how to handle this yet
        next_states["mm"] = None
        next_states["qa"] = None
        next_states["explore"] = None

    def fill_replay_buffer(self) -> None:
        """Make the replay buffer full in the beginning with the uniformly-sampled
        actions. The filling continues until it reaches the warm start size.

        """
        new_episode_starts = True
        while len(self.replay_buffer) < self.warm_start:

            if new_episode_starts:
                self.init_memory_systems()
                observations, info = self.env.reset()
                done = False
                new_episode_starts = False

            observations, action, reward, done, q_values = self.step(
                observations, greedy=False, save_to_replay_buffer=False
            )

            if done:
                new_episode_starts = True

    def train(self) -> None:
        """Code for training"""
        raise NotImplementedError("Should be implemented by the inherited class!")

    def validate(self) -> None:
        self.dqn.eval()
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
        self.dqn.train()

    def test(self, checkpoint: str = None) -> None:
        self.dqn.eval()
        self.env_config["seed"] = self.test_seed
        self.env = gym.make(self.env_str, **self.env_config)

        assert len(self.val_filenames) == 1
        self.dqn.load_state_dict(torch.load(self.val_filenames[0]))
        if checkpoint is not None:
            self.dqn.load_state_dict(torch.load(checkpoint))

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
        self.dqn.train()

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
