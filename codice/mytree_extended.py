import numpy as np

class ExtendedTree:
    def __init__(self, node):
        self.node = node

    def validate_entry(self, node, x):
        if(node.d == 0):
            ++node.count
        elif(np.dot(node.a,x) < node.b):
            self.validate_entry(self,node.leftChild,x)
        elif(np.dot(node.a,x) >= node.b):
            self.validate_entry(self, node.rightChild, x)
