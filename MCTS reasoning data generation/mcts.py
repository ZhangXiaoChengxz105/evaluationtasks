import json
import random
from collections import defaultdict
from typing import List, Dict
import math

def qa_reasoning(example):
    return f"[QA_REASONING] Analyzing Q: {example['question']} with choices {example['choices']}"

def meta_reasoning(example):
    return f"[META_REASONING] Hint: {example['hint']} | Lecture: {example['lecture']}"

def pic_reasoning(example):
    return f"[PIC_REASONING] Using image file: {example['image_path']} with Q: {example['question']}"

def finish_and_answer(example,node): # in this part, we will go through the node and the parent nodes to get the actions taken
    pred = random.choice(example['choices'])
    is_correct = (example['choices'].index(pred) == example['answer'])
    return {
        "answer": pred,
        "correct": is_correct,
    }
    
class MCTSNode:
    def __init__(self,action, parent=None):
        self.trace = [] #trace only contains previous actions
        self.action=action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0   
        temparent = self.parent 
        while temparent!= None:
            if temparent.action != None:
                self.trace.append(temparent.action)
            temparent = temparent.parent
            
            
    def is_terminal(self): # since we are limiting the number of steps to 3, we can just check if the length of the trace is 3
        if len(self.trace) == 2:
            return True
        else:
            return False
            
    def get_available_actions(self): # this is something we can expand when we do more complicated tasks
    #it gets the possible states fromt the current one
        actions = []
        if 'qa' not in self.trace and self.action != 'qa':
            actions.append('qa')
        if 'meta' not in self.trace and self.action != 'meta':
            actions.append('meta')
        if 'pic' not in self.trace and self.action != 'pic':
            actions.append('pic')
        return actions        
        
        

    def expand(self): #this node has never been sampled therefore expand it
        if self.is_terminal():
            return False
        actions = self.get_available_actions()
        for action in actions:
            newnode = MCTSNode(action, self)
            self.children.append(newnode)
    
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
    
def run_mcts(example:Dict, n_rollouuts:int =20):
    root = MCTSNode(None)
        
    for _ in range(n_rollouuts):
        node = root
        while not node.is_terminal():
            if node.visits == 0:
                node.expand()
            node = node.best_child()
        result = finish_and_answer(example, node)
        reward = 1 if result['correct'] else 0
        node.backpropagate(reward)
    return root
        
if __name__ == "__main__":
    with open("scienceqa_mcts_data.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]

    subject_groups = defaultdict(list)
    for item in data:
        subject = item.get("subject") or item.get("metadata", {}).get("subject", "unknown")
        subject_groups[subject].append(item)

    print("Loaded subjects:", list(subject_groups.keys()))
    print("\n==================== TREE OUTPUT ====================\n") 
    for subject, examples in subject_groups.items():
        print(f"=== Subject: {subject} ===")
        example = examples[0]
        root = run_mcts(example)
        result = finish_and_answer(example, root)
        print("Final predicted result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        print("Tree stats (value/visits) per node:")
        def print_tree(node, indent=0):
            prefix = "  " * indent
            print(f"{prefix}Trace={node.trace + ([node.action] if node.action else [])} | V={node.value}, N={node.visits}, A={node.action}")
            for child in node.children:
                print_tree(child, indent + 1)
        print_tree(root)
        print()               
    
        
