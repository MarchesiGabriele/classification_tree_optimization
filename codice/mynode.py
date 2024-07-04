import numpy as np
from collections import deque

class Node:
    def __init__(self, idx, a=None, b=None, d=None, left=None, right=None, type=False):
        self.idx = idx
        self.a = a # split
        self.b = b # split
        self.classname = None # per le foglie
        self.d = d # nodo branch è attivo
        self.leftChild = left 
        self.rightChild = right

def create_complete_tree(idx, depth, current_depth=0):
    if current_depth == depth:
        return None
    node = Node(idx)
    node.isbranch = current_depth < depth - 1
    if node.isbranch:
        node.leftChild = create_complete_tree(2*idx, depth, current_depth + 1)
        node.rightChild = create_complete_tree(2*idx+1, depth, current_depth + 1)
    return node

def print_tree(node, level=0):
    if node is not None:
        print(' ' * 4 * level + '->', f'Branch{node.idx}' if node.isbranch else f'Leaf{node.idx}')
        print_tree(node.leftChild, level + 1)
        print_tree(node.rightChild, level + 1)

def init_bfs(root, avalues, bvalues, dvalues, classnames, depth):
    if root is None:
        return
    queue = deque([root])
    while queue:
        current:Node = queue.popleft() 
        if current.leftChild != None and current.rightChild != None: # se è branch
            current.a = avalues[:, current.idx-1]
            current.b = bvalues[current.idx].value
            current.d = dvalues[current.idx].value 
        else:
            current.classname = classnames[current.idx- (2**depth-1)]
        
        if current.leftChild:
            queue.append(current.leftChild)
        if current.rightChild:
            queue.append(current.rightChild)

def validate(x, yvalue, root:Node) -> int: # 0 -> wrong classfication, 1 -> correct
    node = root
    while node.leftChild != None or node.rightChild != None: # siamo neel branch
        if np.dot(node.a,x) < node.b:
          node = node.leftChild  
        else:
          node = node.rightChild
    if node.classname == yvalue:
        return 1
    return 0


# Esempio di utilizzo
depth = 3
root = create_complete_tree(1, depth)
print_tree(root)




def validate_entry(self, node, x):
    if(node.d == 0):
        ++node.count
    elif(np.dot(node.a,x) < node.b):
        self.validate_entry(self,node.leftChild,x)
    elif(np.dot(node.a,x) >= node.b):
        self.validate_entry(self, node.rightChild, x)


