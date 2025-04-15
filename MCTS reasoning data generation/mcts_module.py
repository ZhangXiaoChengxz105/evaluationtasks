# === mcts_module.py ===
# Core implementation of MCTS for reasoning path search

import math
import random

class State:
    def __init__(self, question, history=None, used_tools=None):
        self.question = question
        self.history = history or []
        self.used_tools = used_tools or []
        self.prev_action = None
        self.finished = False
        self.answer = None

    def get_actions(self):
        actions = ["GEN_THOUGHT"]
        if "RETRIEVER" not in self.used_tools:
            actions.append("USE_RETRIEVER")
        actions.append("FINISH")
        return actions

    def take_action(self, action):
        new_state = State(self.question, list(self.history), list(self.used_tools))
        new_state.prev_action = action

        if action == "GEN_THOUGHT":
            new_state.history.append("<THOUGHT: reasoning step>")
        elif action == "USE_RETRIEVER":
            new_state.history.append("<TOOL: retrieved evidence>")
            new_state.used_tools.append("RETRIEVER")
        elif action == "FINISH":
            new_state.finished = True
            # for simplicity, echo back a fixed answer based on the question
            if "capital of France" in self.question:
                new_state.answer = "Paris"
            elif "3 + 5" in self.question:
                new_state.answer = "8"
            elif "Red Planet" in self.question:
                new_state.answer = "Mars"
            else:
                new_state.answer = "generated answer"
        return new_state

    def is_terminal(self):
        return self.finished

    def reward(self, correct_answer):
        return float(self.answer == correct_answer)

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0

    def ucb_score(self, total_visits, c=1.4):
        if self.visit_count == 0:
            return float('inf')
        exploitation = self.value_sum / self.visit_count
        exploration = c * math.sqrt(math.log(total_visits + 1) / self.visit_count)
        return exploitation + exploration

class MCTSMultimodalQA:
    def __init__(self, root_state, correct_answer, num_sim=30):
        self.root = MCTSNode(root_state)
        self.correct_answer = correct_answer
        self.num_sim = num_sim

    def select(self, node):
        while node.children:
            node = max(node.children, key=lambda n: n.ucb_score(node.visit_count))
        return node

    def expand(self, node):
        actions = node.state.get_actions()
        for action in actions:
            new_state = node.state.take_action(action)
            node.children.append(MCTSNode(new_state, parent=node))

    def simulate(self, state):
        current = state
        for _ in range(5):
            if current.is_terminal():
                break
            action = random.choice(current.get_actions())
            current = current.take_action(action)
        return current.reward(self.correct_answer)

    def backpropagate(self, node, reward):
        while node:
            node.visit_count += 1
            node.value_sum += reward
            node = node.parent

    def run(self):
        for _ in range(self.num_sim):
            leaf = self.select(self.root)
            if not leaf.state.is_terminal():
                self.expand(leaf)
                leaf = random.choice(leaf.children)
            reward = self.simulate(leaf.state)
            self.backpropagate(leaf, reward)

        best = max(self.root.children, key=lambda n: n.visit_count)
        return best.state.history, best.state.answer
