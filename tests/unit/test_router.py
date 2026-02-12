"""Unit tests for teamtr routers."""

import pytest

# Note: These imports will work once dependencies are installed
try:
    from verl.teamtr.router import RandomRouter, RoundRobinRouter
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="verl.teamtr not importable (dependencies missing)")


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
class TestRandomRouter:
    """Test RandomRouter functionality."""

    def test_initialization(self):
        """Test router initializes correctly."""
        router = RandomRouter(num_agents=3)
        assert router.num_agents == 3

    def test_select_agent_range(self):
        """Test that selected agents are in valid range."""
        router = RandomRouter(num_agents=5, seed=42)
        for _ in range(100):
            agent_id = router.select_agent()
            assert 0 <= agent_id < 5

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same sequence."""
        router1 = RandomRouter(num_agents=3, seed=42)
        router2 = RandomRouter(num_agents=3, seed=42)

        sequence1 = [router1.select_agent() for _ in range(10)]
        sequence2 = [router2.select_agent() for _ in range(10)]

        assert sequence1 == sequence2


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Imports not available")
class TestRoundRobinRouter:
    """Test RoundRobinRouter functionality."""

    def test_initialization(self):
        """Test router initializes correctly."""
        router = RoundRobinRouter(num_agents=3)
        assert router.num_agents == 3
        assert router.current_idx == 0

    def test_round_robin_sequence(self):
        """Test that router produces correct round-robin sequence."""
        router = RoundRobinRouter(num_agents=3)
        expected = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        actual = [router.select_agent() for _ in range(9)]
        assert actual == expected

    def test_reset(self):
        """Test that reset() restarts sequence from 0."""
        router = RoundRobinRouter(num_agents=3)
        router.select_agent()  # 0
        router.select_agent()  # 1
        router.reset()
        assert router.select_agent() == 0
