"""Agent that uses a GNN for its DQN, for the RoomEnv2 environment."""

import datetime
import os
from copy import deepcopy
import shutil
from typing import Literal

import gymnasium as gym
import torch
import torch.optim as optim
from humemai.utils import is_running_notebook, write_yaml
from humemai.memory import EpisodicMemory, MemorySystems, SemanticMemory, ShortMemory

from .nn import GNN
from .utils import (
    ReplayBuffer,
    plot_results,
    save_final_results,
    save_states_q_values_actions,
    save_validation,
    select_action,
    target_hard_update,
    update_epsilon,
    update_model,
)
from ..policy import (
    encode_all_observations,
    answer_question,
    manage_memory,
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
            "short": 10,
        },
        pretrain_semantic: str | bool = False,
        gnn_params: dict = {
            "embedding_dim": 128,
            "num_layers_GNN": 4,
            "num_layers_MLP": 4,
        },
        run_test: bool = True,
        num_samples_for_results: int = 10,
        plotting_interval: int = 10,
        train_seed: int = 5,
        test_seed: int = 0,
        device: str = "cpu",
        mm_policy: str = "generalize",
        qa_policy: str = "latest_strongest",
        explore_policy: str = "avoid_walls",
        env_config: dict = {
            "question_prob": 1.0,
            "terminates_at": 99,
            "randomize_observations": "all",
            "room_size": "l",
            "rewards": {"correct": 1, "wrong": 0, "partial": 0},
            "make_everything_static": False,
            "num_total_questions": 1000,
            "question_interval": 1,
            "include_walls_in_observations": True,
        },
        ddqn: bool = True,
        default_root_dir: str = "./training-results/stochastic-objects/DQN/",
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
            pretrain_semantic: whether to pretrain the semantic memory system.
            gnn_params: parameters for the neural network (GNN)
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

        """
        params_to_save = deepcopy(locals())
        del params_to_save["self"]
        self.default_root_dir = os.path.join(
            default_root_dir, str(datetime.datetime.now())
        )
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
        assert self.qa_policy in ["random", "latest_strongest", "strongest_latest"]
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
        self.pretrain_semantic = pretrain_semantic
        self.env = gym.make(self.env_str, **self.env_config)

        self.device = torch.device(device)
        print(f"Running on {self.device}")

        self.ddqn = ddqn
        self.val_filenames = {"best": None, "last": None}
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

        self.action_mm2str = {0: "episodic", 1: "semantic", 2: "forget"}
        self.action_explore2str = {
            0: "north",
            1: "east",
            2: "south",
            3: "west",
            4: "stay",
        }

        self.init_memory_systems()

        self.gnn_params = gnn_params
        self.gnn_params["device"] = self.device
        self.gnn_params["entities"] = [
            e for entities in self.env.unwrapped.entities.values() for e in entities
        ]
        self.gnn_params["relations"] = (
            self.env.unwrapped.relations + self.memory_systems.qualifier_relations
        )
        self.dqn = GNN(**self.gnn_params)
        self.dqn_target = GNN(**self.gnn_params)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.replay_buffer_mm = ReplayBuffer(self.replay_buffer_size, self.batch_size)
        self.replay_buffer_explore = ReplayBuffer(replay_buffer_size, batch_size)

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        self.q_values = {
            "train": {"mm": [], "explore": []},
            "val": {"mm": [], "explore": []},
            "test": {"mm": [], "explore": []},
        }

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
            episodic=EpisodicMemory(capacity=self.capacity["episodic"]),
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

    def get_deepcopied_working_memory(self) -> list[list]:
        """Get a deepcopied memory state.

        This is necessary because the memory state is a list of lists, which is
        mutable.

        Returns:
            deepcopied working memory

        """
        return deepcopy(self.memory_systems.working.to_list())

    def move_agent_to_episodic_memory(self) -> None:
        """Move the agent's location related short-term memories to the episodic memory
        system."""
        for mem_short in self.memory_systems.short:
            if mem_short[0] == "agent":
                manage_memory(self.memory_systems, "episodic", mem_short)

    def step_a(
        self,
        greedy: bool,
        save_q_mm: bool,
        train_val_test: Literal["train", "val", "test"] | None = None,
    ) -> None:
        """Step a of the algorithm.

        $\pi_{mm}$:
            IN:  [    ,     ,     ,     ,     ]    OUT: [s   , a   ,     ,     ,     ]
        $\pi_{expore}$:
            IN:  [    ,     ,     ,     ,     ]    OUT: [    ,     ,     ,     ,     ]
        $\pi_{qa}$:
            IN:  [    ,     ,     ,     ,     ]    OUT: [    ,     ,     ,     ,     ]

        blanks are Nones

        Args:
            greedy: whether to use greedy policy
            save_q_mm: whether to save q_values
            train_val_test: whether to train, validate, or test

        """
        self.init_memory_systems()
        observations, info = self.env.reset()
        self.observations = observations["room"]
        self.questions = observations["questions"]
        encode_all_observations(self.memory_systems, self.observations)
        self.move_agent_to_episodic_memory()

        # mm
        s_mm = self.get_deepcopied_working_memory()
        a_mm, q_mm = select_action(  # the dimension of a_mm is [num_actions_taken]
            state=s_mm,
            greedy=greedy,
            dqn=self.dqn,
            epsilon=self.epsilon,
            policy_type="mm",
        )
        assert len(a_mm) == len(self.memory_systems.short)
        for a_mm_, mem_short in zip(a_mm, self.memory_systems.short):
            manage_memory(self.memory_systems, self.action_mm2str[a_mm_], mem_short)

        self.tuple_mm = [s_mm, a_mm, None, None, None]

        if save_q_mm:
            for q_mm_ in q_mm:
                self.q_values[train_val_test]["mm"].append(q_mm_)

        return s_mm, q_mm, a_mm

    def step_b(
        self,
        greedy: bool,
        save_mm_to_replay_buffer: bool,
        save_q_explore: bool,
        train_val_test: Literal["train", "val", "test"] | None = None,
    ) -> tuple[float, bool]:
        """Step b of the algorithm.

        $\pi_{mm}$:
            IN:  [s   , a   ,     ,     ,     ]    OUT: [s   , a   , r   , s'  , done]
        $\pi_{expore}$:
            IN:  [    ,     ,     ,     ,     ]    OUT: [s   , a   , r   ,     ,     ]
        $\pi_{qa}$:
            IN:  [    ,     ,     ,     ,     ]    OUT: [s   , a   , r   ,     ,     ]

        blanks are Nones

        Out of the three (a, b, and c) steps, this is the only step that interacts
        with the environment.

        Args:
            greedy: whether to use greedy policy
            save_mm_to_replay_buffer: whether to save to replay buffer
            save_q_explore: whether to save q_values
            train_val_test: whether to train, validate, or test

        Returns:
            reward: reward from the environment
            done: whether the episode is done

        """
        assert self.tuple_mm[2:] == [None, None, None]

        # explore
        s_explore = self.get_deepcopied_working_memory()

        a_explore, q_explore = select_action(
            state=s_explore,
            greedy=greedy,
            dqn=self.dqn,
            epsilon=self.epsilon,
            policy_type="explore",
        )

        # qa
        a_qa = [
            answer_question(
                self.memory_systems,
                self.qa_policy,
                question,
            )
            for question in self.questions
        ]

        action_pair = (a_qa, self.action_explore2str[a_explore[0]])
        (
            observations,
            reward,
            done,
            truncated,
            info,
        ) = self.env.step(action_pair)
        done = done or truncated

        self.observations = observations["room"]
        self.questions = observations["questions"]
        encode_all_observations(self.memory_systems, self.observations)
        self.move_agent_to_episodic_memory()

        # mm
        s_next_mm = self.get_deepcopied_working_memory()

        self.tuple_mm[2] = reward
        self.tuple_mm[3] = s_next_mm
        self.tuple_mm[4] = done

        self.tuple_explore = [s_explore, None, None, None, None]
        self.tuple_explore[1] = a_explore
        self.tuple_explore[2] = reward

        if save_mm_to_replay_buffer:
            self.replay_buffer_mm.store(*self.tuple_mm)

        if save_q_explore:
            for q_explore_ in q_explore:
                self.q_values[train_val_test]["explore"].append(q_explore_)

        return s_explore, q_explore, a_explore, reward, done

    def step_c(
        self,
        greedy: bool,
        save_explore_to_replay_buffer: bool,
        done: bool,
        save_q_mm: bool,
        train_val_test: Literal["train", "val", "test"] | None = None,
    ) -> None:
        """Step c of the algorithm.

        $\pi_{mm}$:
            IN:  [    ,     ,     ,     ,     ]    OUT: [s   , a   ,     ,     ,     ]
        $\pi_{expore}$:
            IN:  [s   , a   , r   ,     ,     ]    OUT: [s   , a   , r   , s'  , done]
        $\pi_{qa}$:
            IN:  [s   , a   , r   ,     ,     ]    OUT: [s   , a   , r   , s'  , done]

        blanks are Nones

        Args:
            greedy: whether to use greedy policy
            save_explore_to_replay_buffer: whether to save to replay buffer
            done: whether the episode was done
            save_q_mm: whether to save q_values
            train_val_test: whether to train, validate, or test

        """
        assert self.tuple_explore[3:] == [None, None]

        s_mm = self.get_deepcopied_working_memory()

        # mm
        a_mm, q_mm = select_action(
            state=s_mm,
            greedy=greedy,
            dqn=self.dqn,
            epsilon=self.epsilon,
            policy_type="mm",
        )

        # the dimension of a_mm is [num_actions_taken]
        assert len(a_mm) == len(self.memory_systems.short)
        for a_mm_, mem_short in zip(a_mm, self.memory_systems.short):
            manage_memory(self.memory_systems, self.action_mm2str[a_mm_], mem_short)

        # explore
        s_next_explore = self.get_deepcopied_working_memory()

        self.tuple_mm = [s_mm, a_mm, None, None, None]
        self.tuple_explore[3] = s_next_explore
        self.tuple_explore[4] = done

        if save_explore_to_replay_buffer:
            self.replay_buffer_explore.store(*self.tuple_explore)

        if save_q_mm:
            for q_mm_ in q_mm:
                self.q_values[train_val_test]["mm"].append(q_mm_)

        return s_mm, q_mm, a_mm

    def fill_replay_buffer(self) -> None:
        """Make the replay buffer full in the beginning with the uniformly-sampled
        actions. The filling continues until it reaches the warm start size.

        """
        new_episode_starts = True
        while (
            len(self.replay_buffer_mm) < self.warm_start
            or len(self.replay_buffer_explore) < self.warm_start
        ):
            if new_episode_starts:
                s_mm, q_mm, a_mm = self.step_a(
                    greedy=False,
                    save_q_mm=False,
                    train_val_test=None,
                )
                new_episode_starts = False

            s_explore, q_explore, a_explore, reward, done = self.step_b(
                greedy=False,
                save_mm_to_replay_buffer=True,
                save_q_explore=False,
                train_val_test=None,
            )
            s_mm, q_mm, a_mm = self.step_c(
                greedy=False,
                save_explore_to_replay_buffer=True,
                done=done,
                save_q_mm=False,
                train_val_test=None,
            )

            if done:
                new_episode_starts = True

    def train(self) -> None:
        """Train the agent."""
        self.fill_replay_buffer()  # fill up the buffer till warm start size
        self.num_validation = 0

        self.epsilons = []
        self.training_loss = {"total": [], "mm": [], "explore": []}
        self.scores = {"train": [], "val": [], "test": None}

        self.dqn.train()

        new_episode_starts = True
        score = 0
        self.iteration_idx = 0

        while True:
            if new_episode_starts:
                s_mm, q_mm, a_mm = self.step_a(
                    greedy=False,
                    save_q_mm=True,
                    train_val_test="train",
                )
                new_episode_starts = False

            s_explore, q_explore, a_explore, reward, done = self.step_b(
                greedy=False,
                save_mm_to_replay_buffer=True,
                save_q_explore=True,
                train_val_test="train",
            )
            s_mm, q_mm, a_mm = self.step_c(
                greedy=False,
                save_explore_to_replay_buffer=True,
                done=done,
                save_q_mm=True,
                train_val_test="train",
            )
            score += reward
            self.iteration_idx += 1

            if done:
                new_episode_starts = True
                self.scores["train"].append(score)
                score = 0
                with torch.no_grad():
                    self.validate()

            if not new_episode_starts:
                loss_mm, loss_explore, loss = update_model(
                    replay_buffer_mm=self.replay_buffer_mm,
                    replay_buffer_explore=self.replay_buffer_explore,
                    optimizer=self.optimizer,
                    device=self.device,
                    dqn=self.dqn,
                    dqn_target=self.dqn_target,
                    ddqn=self.ddqn,
                    gamma=self.gamma,
                )

                self.training_loss["total"].append(loss)
                self.training_loss["mm"].append(loss_mm)
                self.training_loss["explore"].append(loss_explore)

                # linearly decay epsilon
                self.epsilon = update_epsilon(
                    self.epsilon,
                    self.max_epsilon,
                    self.min_epsilon,
                    self.epsilon_decay_until,
                )
                self.epsilons.append(self.epsilon)

                # if hard update is needed
                if self.iteration_idx % self.target_update_interval == 0:
                    target_hard_update(self.dqn, self.dqn_target)

                # plotting & show training results
                if (
                    self.iteration_idx == self.num_iterations
                    or self.iteration_idx % self.plotting_interval == 0
                ):
                    self.plot_results("all", save_fig=True)

                if self.iteration_idx == self.num_iterations:
                    break

        with torch.no_grad():
            self.test()

        self.env.close()

    def validate_test_middle(self, val_or_test: str) -> tuple[list, list, list, list]:
        """A function shared by explore validation and test in the middle.

        Args:
            val_or_test: "val" or "test"

        Returns:
            scores_local: a list of total episode rewards
            states_local: memory states
            q_values_local: q values
            actions_local: greey actions taken

        """
        scores_local = []
        states_local = []
        q_values_local = []
        actions_local = []

        for idx in range(self.num_samples_for_results):
            new_episode_starts = True
            score = 0
            if idx == self.num_samples_for_results - 1:
                save_useful = True
            else:
                save_useful = False

            while True:
                if new_episode_starts:
                    s_mm, q_mm, a_mm = self.step_a(
                        greedy=True,
                        save_q_mm=save_useful,
                        train_val_test=val_or_test,
                    )
                    new_episode_starts = False

                    if save_useful:
                        state = {"mm": s_mm}
                        q_values = {"mm": q_mm}
                        action = {"mm": a_mm}

                s_explore, q_explore, a_explore, reward, done = self.step_b(
                    greedy=True,
                    save_mm_to_replay_buffer=False,
                    save_q_explore=save_useful,
                    train_val_test=val_or_test,
                )

                if save_useful:
                    state["explore"] = s_explore
                    q_values["explore"] = q_explore
                    action["explore"] = a_explore

                    states_local.append(state)
                    q_values_local.append(q_values)
                    actions_local.append(action)

                s_mm, q_mm, a_mm = self.step_c(
                    greedy=True,
                    save_explore_to_replay_buffer=False,
                    done=done,
                    save_q_mm=save_useful,
                    train_val_test=val_or_test,
                )
                score += reward

                if save_useful:
                    state = {"mm": s_mm}
                    q_values = {"mm": q_mm}
                    action = {"mm": a_mm}

                if done:
                    break

            scores_local.append(score)

        return scores_local, states_local, q_values_local, actions_local

    def validate(self) -> None:
        """Validate the agent."""
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
        """Test the agent."""
        self.dqn.eval()
        self.env_config["seed"] = self.test_seed
        self.env = gym.make(self.env_str, **self.env_config)

        self.dqn.load_state_dict(torch.load(self.val_filenames["best"]))
        if checkpoint is not None:
            self.dqn.load_state_dict(torch.load(checkpoint))

        scores, states, q_values, actions = self.validate_test_middle("test")
        self.scores["test"] = scores

        save_final_results(
            self.scores,
            self.training_loss,
            self.default_root_dir,
            self.q_values,
            self,
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
            self.num_iterations,
            self.env.unwrapped.total_maximum_episode_rewards,
            self.default_root_dir,
            to_plot,
            save_fig,
        )
