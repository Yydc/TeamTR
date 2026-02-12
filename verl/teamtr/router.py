# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Agent routers for multi-agent orchestration."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod


class BaseRouter(ABC):
    """Base class for agent routers."""

    @abstractmethod
    def select_agent(self, context: str = "") -> int:
        """Select next agent ID given conversation context.

        Args:
            context: Optional conversation history for context-aware routing

        Returns:
            Agent ID (0-indexed integer)
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset router state for new conversation."""
        pass


class RandomRouter(BaseRouter):
    """Randomly selects agents with uniform probability."""

    def __init__(self, num_agents: int, seed: int | None = None):
        """Initialize random router.

        Args:
            num_agents: Number of agents in the team
            seed: Optional random seed for reproducibility
        """
        self.num_agents = num_agents
        self.rng = random.Random(seed)

    def select_agent(self, context: str = "") -> int:
        """Select agent uniformly at random.

        Args:
            context: Ignored for random selection

        Returns:
            Random agent ID in [0, num_agents)
        """
        return self.rng.randrange(self.num_agents)

    def reset(self):
        """Reset has no effect for stateless random router."""
        pass


class RoundRobinRouter(BaseRouter):
    """Selects agents in round-robin order."""

    def __init__(self, num_agents: int):
        """Initialize round-robin router.

        Args:
            num_agents: Number of agents in the team
        """
        self.num_agents = num_agents
        self.current_idx = 0

    def select_agent(self, context: str = "") -> int:
        """Select next agent in round-robin order.

        Args:
            context: Ignored for round-robin

        Returns:
            Next agent ID in sequence
        """
        agent_id = self.current_idx
        self.current_idx = (self.current_idx + 1) % self.num_agents
        return agent_id

    def reset(self):
        """Reset round-robin counter to start from agent 0."""
        self.current_idx = 0
