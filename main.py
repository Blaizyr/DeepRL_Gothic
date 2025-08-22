# main.py
from testing.test_runner import TestRunner
if __name__ == "__main__":
    runner = TestRunner("all_test_scenarios.yml")
    runner.run()
