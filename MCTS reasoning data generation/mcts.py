import json
import random
from collections import defaultdict
from typing import List, Dict
import math
from reasonings import qa_reasoning, meta_reasoning, pic_reasoning


def finish_and_answer(example,node): 
    # in this part, we will go through the node and the parent nodes to get the actions taken
    # for routes that yield the correct answer sooner, we have higher reward
    # for routes that do not yield the correct answer, we have zero as the reaward
    trace_sequence = node.trace + ([node.action] if node.action else [])
    reward = 0
    tempexample = example.copy()
    corrected = False # this is a flag to check if we have already got the correct answer
    for i,action in enumerate(trace_sequence):
        if action == 'qa':
            tempexample,answer = qa_reasoning(tempexample)
            if answer == example['answer'] and (corrected == False):
                corrected = True    
                reward += 2**(2-i)
        elif action == 'meta':
            tempexample,answer = meta_reasoning(tempexample)
            if answer == example['answer'] and (corrected == False):
                corrected = True
                reward += 2**(2-i)
        elif action == 'pic':
            tempexample,answer = pic_reasoning(tempexample)
            if answer == example['answer'] and (corrected == False):
                corrected = True
                reward += 2**(2-i)
    
    return reward
    
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
        self.trace.reverse()
            
            
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
        node.backpropagate(result)
    return root
        
if __name__ == "__main__":
    with open("scienceqa_mcts_data.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
        example = json.loads(lines[1])

    print("\n==================== TREE OUTPUT ====================\n")
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
    
        

        
