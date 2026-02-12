#!/bin/bash
# TeamTR Test Runner
#
# Usage:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh unit         # Run only unit tests
#   ./run_tests.sh -k router    # Run tests matching pattern
#   ./run_tests.sh -v           # Verbose output

set -e

# Change to project root
cd "$(dirname "$0")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "TeamTR Test Suite"
echo "=========================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${YELLOW}WARNING: pytest not installed${NC}"
    echo "Installing pytest..."
    pip install pytest pytest-cov
fi

# Determine test target
if [ $# -eq 0 ]; then
    echo "Running all tests..."
    pytest tests/
elif [ "$1" == "unit" ]; then
    echo "Running unit tests only..."
    pytest tests/unit/
else
    echo "Running tests with arguments: $@"
    pytest "$@"
fi

# Capture exit code
TEST_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
fi
echo "=========================================="

exit $TEST_EXIT_CODE
