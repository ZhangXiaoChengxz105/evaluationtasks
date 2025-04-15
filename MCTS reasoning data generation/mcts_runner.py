# === mcts_runner.py ===
# Load ScienceQA data, split by subject, and construct mock MCTS search trees per subject

import json
import random
from collections import defaultdict
from typing import List, Dict
import math

# --- Step Simulators (mock reasoning tool calls) ---
def qa_reasoning(example):
    return f"[QA_REASONING] Analyzing Q: {example['question']} with choices {example['choices']}"

def meta_reasoning(example):
    return f"[META_REASONING] Hint: {example['hint']} | Lecture: {example['lecture']}"

def pic_reasoning(example):
    return f"[PIC_REASONING] Using image file: {example['image_path']} with Q: {example['question']}"

def finish_and_answer(example, steps_trace):
    pred = random.choice(example['choices'])
    is_correct = (example['choices'].index(pred) == example['answer'])
    return {
        "answer": pred,
        "correct": is_correct,
        "trace": steps_trace
    }

# --- MCTS Tree Node ---
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.action= None
        self.visits = 0
        self.value = 0

    def is_terminal(self):
        return self.state.finished

    def expand(self):
        actions = self.state.get_available_actions()
        if not actions:
            next_state = self.state.copy()
            next_state.finished = True 
            
        for action in actions:
            next_state = self.state.copy()
            result = next_state.apply_action(action)
            if result:
                next_state.result = result
            self.children.append(MCTSNode(next_state, self))

    def best_child(self, c_param=1.4):
        def ucb_score(child):
            if child.visits == 0:
                return float('inf')
            exploit = child.value / child.visits
            explore = c_param * math.sqrt(math.log(self.visits + 1) / child.visits)
            return exploit + explore
        return max(self.children, key=ucb_score)

    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

# --- MCTS State Representation ---
class MCTSState:
    def __init__(self, example):
        self.example = example
        self.used_steps = set()
        self.trace = []
        self.finished = False
        self.result = None

    def get_available_actions(self): # finish is also one kind of action
        actions = []
        if 'QA_REASONING' not in self.used_steps:
            actions.append('QA_REASONING')
        if 'META_REASONING' not in self.used_steps and self.example.get('hint'):
            actions.append('META_REASONING')
        if 'PIC_REASONING' not in self.used_steps and self.example.get('image_path'):
            actions.append('PIC_REASONING')
        # actions.append('FINISH')
        return actions

    def apply_action(self, action):
        self.used_steps.add(action)
        if action == 'QA_REASONING':
            obs = qa_reasoning(self.example)
        elif action == 'META_REASONING':
            obs = meta_reasoning(self.example)
        elif action == 'PIC_REASONING':
            obs = pic_reasoning(self.example)
        elif action == 'FINISH':
            self.finished = True
            self.result = finish_and_answer(self.example, self.trace)
            return self.result
        else:
            obs = f"[Unknown step {action}]"
        self.trace.append(f"{action}: {obs}")
        return None

    def copy(self):
        new_state = MCTSState(self.example)
        new_state.used_steps = set(self.used_steps)
        new_state.trace = list(self.trace)
        new_state.finished = self.finished
        return new_state

# --- MCTS Simulation per subject ---
def run_mcts(example: Dict, n_rollouts: int = 20) -> MCTSNode:
    root = MCTSNode(MCTSState(example))

    for _ in range(n_rollouts):
        node = root
        # Selection and expansion
        while node.children:
            node = node.best_child()
        if not node.state.finished:
            node.expand()
            if node.children:
                node = node.best_child()
        # Simulation (simulate on a copy of the state to avoid side effects)
        sim_state = node.state.copy()
        for step in ["QA_REASONING", "META_REASONING", "PIC_REASONING"]:
            if step not in sim_state.used_steps:
                sim_state.apply_action(step)
        result = sim_state.apply_action("FINISH")
        if result is None:
            result = finish_and_answer(example, sim_state.trace)
        node.backpropagate(int(result['correct']))

    return root

# --- Main execution ---
if __name__ == "__main__":
    with open("scienceqa_mcts_data.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]

    # Split by subject
    subject_groups = defaultdict(list)
    for item in data:
        subject = item['metadata']['subject'] or 'unknown'
        subject_groups[subject].append(item)

    print("Loaded subjects:", list(subject_groups.keys()))
    print("==================== TREE OUTPUT ====================")

    # Run MCTS for first 3 examples per subject
    for subject, examples in subject_groups.items():
        print(f"=== Subject: {subject} ===")
        for i, ex in enumerate(examples[:3]):
            print(f"--- [{subject}] Example {i} ---")
            root = run_mcts(ex)
            result = root.state.result
            # Print final root result after 20 rollouts
            print(json.dumps(root.state.result or {}, indent=2, ensure_ascii=False))
            print("Tree stats (value/visits) per node:")
            def print_tree(node, indent=0):
                print("  " * indent + f"Trace={node.state.trace[-1] if node.state.trace else 'ROOT'} | V={node.value}, N={node.visits}")
                for child in node.children:
                    print_tree(child, indent + 1)
            # Use final updated root node for printing
            print_tree(root)
            print()
        print(json.dumps(result, indent=2, ensure_ascii=False))
