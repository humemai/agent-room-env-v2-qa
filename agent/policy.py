"""Handcrafted / trained policies.

The trained neural network policies are not implemented yet.
"""

import random
from typing import Literal

import numpy as np
import torch

from humemai.memory import EpisodicMemory, MemorySystems, SemanticMemory, ShortMemory
from .utils import argmax


def encode_observation(memory_systems: MemorySystems, obs: list[str | int]) -> None:
    """Non RL policy of encoding an observation into a short-term memory.

    At the moment, observation is the same as short-term memory, since this is made for
    RoomEnv-v2.

    Args:
        MemorySystems
        obs: observation as a quadruple: [head, relation, tail, int]

    """
    mem_short = ShortMemory.ob2short(obs)
    memory_systems.short.add(mem_short)
    memory_systems.working.update()


def encode_all_observations(
    memory_systems: MemorySystems, obs_multiple: list[list[str | int]]
) -> None:
    """Non RL policy of encoding all observations into short-term memories.

    Args:
        MemorySystems
        obs_multiple: a list of observations

    """
    assert isinstance(obs_multiple, list), "`obs_multi1ple` should be a list."
    for obs in obs_multiple:
        encode_observation(memory_systems, obs)


def find_agent_current_location(memory_systems: MemorySystems) -> str:
    """Find the current location of the agent.

    looks up the episodic. If fails, it looks up the semantic.
    If all fails, it returns None.

    Args:
        MemorySystems

    Returns:
        agent_current_location: str

    """
    if hasattr(memory_systems, "episodic"):
        mems = [
            mem
            for mem in memory_systems.episodic
            if mem[0] == "agent" and mem[1] == "atlocation"
        ]
        if len(mems) > 0:
            agent_current_location = mems[-1][2]
            return agent_current_location

    if hasattr(memory_systems, "semantic"):
        mems = [
            mem
            for mem in memory_systems.semantic
            if mem[0] == "agent" and mem[1] == "atlocation"
        ]
        if len(mems) > 0:
            agent_current_location = mems[-1][2]
            return agent_current_location

    return None


def find_visited_locations(
    memory_systems: MemorySystems,
) -> dict[str, list[list[str, int]]]:
    """Find the locations that the agent has visited so far.

    Args:
        MemorySystems: MemorySystems

    Returns:
        visited_locations: a dictionary of a list of [location, time/strength] pairs.

    """
    visited_locations = {"episodic": [], "semantic": []}

    for mem in memory_systems.episodic:
        if mem[0] == "agent" and mem[1] == "atlocation":
            pair = [mem[2], mem[3]]
            visited_locations["episodic"].append(pair)

    # ascending order
    sorted(visited_locations["episodic"], key=lambda x: x[1])

    for mem in memory_systems.semantic:
        if mem[0] == "agent" and mem[1] == "atlocation":
            pair = [mem[2], mem[3]]
            visited_locations["semantic"].append(pair)

    # ascending order
    sorted(visited_locations["semantic"], key=lambda x: x[1])

    return visited_locations


def explore(
    memory_systems: MemorySystems,
    explore_policy: str,
) -> str:
    """Explore the room (sub-graph).

    Args:
        memory_systems: MemorySystems
        explore_policy: "random" or "avoid_walls"

    Returns:
        action: The exploration action to take.

    """
    if explore_policy == "random":
        action = random.choice(["north", "east", "south", "west", "stay"])
    elif explore_policy == "avoid_walls":
        agent_current_location = find_agent_current_location(memory_systems)

        # no information about the agent's location
        if agent_current_location is None:
            action = random.choice(["north", "east", "south", "west", "stay"])

        # Get all the memories related to the current location
        mems = []

        # from the semantic memory
        if hasattr(memory_systems, "semantic"):
            mems += [
                mem
                for mem in memory_systems.semantic
                if mem[0] == agent_current_location
                and mem[1] in ["north", "east", "south", "west"]
            ]

        # from the episodic
        if hasattr(memory_systems, "episodic"):
            mems += [
                mem
                for mem in memory_systems.episodic
                if mem[0] == agent_current_location
                and mem[1] in ["north", "east", "south", "west"]
            ]

        # we know the agent's current location but there is no memory about the map
        if len(mems) == 0:
            action = random.choice(["north", "east", "south", "west", "stay"])

        else:
            # we know the agent's current location and there is at least one memory
            # about the map and we want to avoid the walls

            to_take = []
            to_avoid = []

            for mem in mems:
                if mem[2].split("_")[0] == "room":
                    to_take.append(mem[1])
                elif mem[2] == "wall":
                    if mem[1] not in to_avoid:
                        to_avoid.append(mem[1])

            if len(to_take) > 0:
                action = random.choice(to_take)
            else:
                options = ["north", "east", "south", "west", "stay"]
                for e in to_avoid:
                    options.remove(e)

                action = random.choice(options)

    else:
        raise ValueError("Unknown exploration policy.")

    assert action in ["north", "east", "south", "west", "stay"]

    return action


def manage_memory(
    memory_systems: MemorySystems,
    policy: str,
    mem_short: list | None = None,
) -> None:
    """Non RL memory management policy.

    Args:
        MemorySystems
        policy: "episodic", "semantic", "generalize", "forget", or "random"
        mem_short: a short-term memory to be moved into a long-term memory. If None,
            then the first element is used.
        mm_policy_model: a neural network model for memory management policy.
        mm_policy_model_type: depends wheter your RL algorithm used.

    """

    def action_number_0():
        assert hasattr(memory_systems, "episodic")
        mem_epi = ShortMemory.short2epi(mem_short)
        check, error_msg = memory_systems.episodic.can_be_added(mem_epi)
        if check:
            memory_systems.episodic.add(mem_epi)
        else:
            if error_msg == "The memory system is full!":
                memory_systems.episodic.forget_oldest()
                memory_systems.episodic.add(mem_epi)
            else:
                raise ValueError(error_msg)

    def action_number_1():
        assert hasattr(memory_systems, "semantic")
        mem_sem = ShortMemory.short2sem(mem_short)
        check, error_msg = memory_systems.semantic.can_be_added(mem_sem)
        if check:
            memory_systems.semantic.add(mem_sem)
        else:
            if error_msg == "The memory system is full!":
                memory_systems.semantic.forget_weakest()
                memory_systems.semantic.add(mem_sem)
            else:
                raise ValueError(error_msg)

    assert memory_systems.short.has_memory(mem_short)
    assert not memory_systems.short.is_empty
    assert policy.lower() in [
        "episodic",
        "semantic",
        "forget",
    ]

    if policy.lower() == "episodic":
        action_number_0()

    elif policy.lower() == "semantic":
        action_number_1()

    elif policy.lower() == "forget":
        pass

    else:
        raise ValueError

    memory_systems.short.forget(mem_short)
    memory_systems.working.update()


def answer_question(
    memory_systems: MemorySystems,
    policy: str,
    question: list[str | int],
) -> None | str | int:
    """Use the memory systems to answer a question from RoomEnv-v2. It assumes that
    the question is a one-hop query. As we closely follow the cognitive science,
    we only use the working memory answer the questions. The working memory is
    short-term + (partial) long-term memory. The long-term memory used is partial,
    since it can be too big.

    Args:
        MemorySystems
        qa_policy: "random", "latest_strongest", or "strongest_latest".
        question: A quadruple given by RoomEnv-v2, e.g., [laptop, atlocation, ?,
            current_time]

    Returns:
        pred: prediction

    """

    def get_mem_with_highest_timestamp(memories: list):
        highest_timestamp_list = None
        highest_timestamp = float("-inf")  # Initialize with negative infinity

        for sublist in memories:
            timestamps = sublist[3]["timestamp"]
            max_timestamp = max(
                timestamps
            )  # Get the maximum timestamp in the current sublist
            if max_timestamp > highest_timestamp:
                highest_timestamp = max_timestamp
                highest_timestamp_list = sublist

        return highest_timestamp_list

    def get_mem_with_highest_strength(memories: list):
        highest_strength_list = None
        highest_strength = float("-inf")  # Initialize with negative infinity

        for sublist in memories:
            strength = sublist[3]["strength"]
            if strength > highest_strength:
                highest_strength = strength
                highest_strength_list = sublist

        return highest_strength_list

    query_idx = question.index("?")
    memories = memory_systems.working.query_memory(question[:-1] + ["?"])

    if len(memories) == 0:
        return None

    if policy.lower() == "random":
        return random.choice(memories)[query_idx]

    elif policy.lower() == "latest_strongest":
        memories_time = [
            mem[:-1] + [{"timestamp": mem[-1]["timestamp"]}]
            for mem in memories
            if "timestamp" in list(mem[-1].keys())
        ]
        memories_strength = [
            mem[:-1] + [{"strength": mem[-1]["strength"]}]
            for mem in memories
            if "strength" in list(mem[-1].keys())
        ]
        if len(memories_time) == 0 and len(memories_strength) == 0:
            return None
        elif len(memories_time) == 0:
            return get_mem_with_highest_strength(memories_strength)[query_idx]
        else:
            return get_mem_with_highest_timestamp(memories_time)[query_idx]

    elif policy.lower() == "strongest_latest":
        memories_time = [
            mem[:-1] + [{"timestamp": mem[-1]["timestamp"]}]
            for mem in memories
            if "timestamp" in list(mem[-1].keys())
        ]
        memories_strength = [
            mem[:-1] + [{"strength": mem[-1]["strength"]}]
            for mem in memories
            if "strength" in list(mem[-1].keys())
        ]
        if len(memories_time) == 0 and len(memories_strength) == 0:
            return None
        elif len(memories_strength) == 0:
            return get_mem_with_highest_timestamp(memories_time)[query_idx]
        else:
            return get_mem_with_highest_strength(memories_strength)[query_idx]

    else:
        raise ValueError("Unknown policy.")
