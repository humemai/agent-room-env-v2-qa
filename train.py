"""Run training for DQN agent with specific parameters."""

import argparse

from agent.dqn import DQNAgent


def main(test_seed, capacity_max) -> None:
    """Run training.

    Args:
        test_seed (int): Seed for testing.
        capacity_max (int): Maximum capacity for the agent.
    """
    room_size = "xl-different-prob"
    terminates_at = 99
    num_iterations = (terminates_at + 1) * 1
    replay_buffer_size = 1
    batch_size = 1
    semantic_decay_factor = 0.8
    num_layers = 2
    triple_qual_weight = 0.8
    embedding_dim = 64
    target_update_interval = 10

    prob_type = (
        "non-equal-object-probs"
        if "different-prob" in room_size
        else "equal-object-probs"
    )
    root_path = (
        f"./training-results/{prob_type}/dqn/"
        f"room_size={room_size}/capacity={capacity_max}/"
    )
    if capacity_max == 192:
        pretrained_path = "trained-results/non-equal-object-probs/dqn/room_size=xl-different-prob/capacity=192/2024-08-12 12:58:16.107541/"
    elif capacity_max == 96:
        pretrained_path = "trained-results/non-equal-object-probs/dqn/room_size=xl-different-prob/capacity=96/2024-08-12 23:58:06.290168/"
    elif capacity_max == 48:
        pretrained_path = "trained-results/non-equal-object-probs/dqn/room_size=xl-different-prob/capacity=48/2024-08-11 11:07:00.648864/"
    elif capacity_max == 24:
        pretrained_path = "trained-results/non-equal-object-probs/dqn/room_size=xl-different-prob/capacity=24/2024-08-11 13:36:54.499426/"
    elif capacity_max == 12:
        pretrained_path = "trained-results/non-equal-object-probs/dqn/room_size=xl-different-prob/capacity=12/2024-08-11 16:24:54.492650/"
    else:
        raise ValueError(f"Invalid capacity_max: {capacity_max}")

    for pretrain_semantic in [False]:
        params_dict = {
            "env_str": "room_env:RoomEnv-v2",
            "num_iterations": num_iterations,
            "replay_buffer_size": replay_buffer_size,
            "warm_start": batch_size,
            "batch_size": batch_size,
            "target_update_interval": target_update_interval,
            "epsilon_decay_until": num_iterations,
            "max_epsilon": 1.0,
            "min_epsilon": 0.1,
            "gamma": {"mm": 0.90, "explore": 0.90},
            "learning_rate": 0.001,
            "capacity": {"long": capacity_max, "short": 15},
            "pretrain_semantic": pretrain_semantic,
            "semantic_decay_factor": semantic_decay_factor,
            "dqn_params": {
                "gcn_layer_params": {
                    "type": "stare",
                    "embedding_dim": embedding_dim,
                    "num_layers": num_layers,
                    "gcn_drop": 0.1,
                    "triple_qual_weight": triple_qual_weight,
                },
                "relu_between_gcn_layers": True,
                "dropout_between_gcn_layers": False,
                "mlp_params": {
                    "num_hidden_layers": num_layers,
                    "dueling_dqn": True,
                },
            },
            "num_samples_for_results": {"val": 1, "test": 10},
            "validation_interval": 1,
            "plotting_interval": 50,
            "train_seed": test_seed + 5,
            "test_seed": test_seed,
            "device": "cpu",
            "env_config": {
                "question_prob": 1.0,
                "terminates_at": terminates_at,
                "randomize_observations": "all",
                "room_size": room_size,
                "rewards": {"correct": 1, "wrong": 0, "partial": 0},
                "make_everything_static": False,
                "num_total_questions": 1000,
                "question_interval": 1,
                "include_walls_in_observations": True,
            },
            "intrinsic_explore_reward": 0,
            "ddqn": True,
            "default_root_dir": root_path,
            "explore_policy": "neural",
            "mm_policy": "neural",
            "qa_function": "llm",
            "pretrained_path": pretrained_path,
            "llm_params": {
                "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "quantization": "4bit",
                "max_new_tokens": 32,
            },
            "scale_reward": False,
        }

        agent = DQNAgent(**params_dict)
        agent.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DQN agent with specific parameters."
    )
    parser.add_argument(
        "--test_seed", type=int, required=True, help="Seed for testing."
    )
    parser.add_argument(
        "--capacity_max",
        type=int,
        required=True,
        help="Maximum capacity for the agent.",
    )

    args = parser.parse_args()

    main(args.test_seed, args.capacity_max)
