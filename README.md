Decision Tree through the ID3 algorithm

Dataset:
Toy Play Tennis dataset as described by Saed Sayad: https://www.saedsayad.com/decision_tree.htm  
Discrete attributes only.  

## Purpose
Gain familiarity with the algorithm by developing it from scratch.  
Hence, best ML practices such as train/test/cross-validation splits 
are NOT prioritized.  

# Run
python Runner.py  

## Implementation
Queue Implementation of the ID3 algorithm

## Results
Accuracy: 0.7857  
It does not seem that ID3 would work for anything but very label-balanced datasets.  
Once an attribute is picked, it can not be reused again. When we ran out of attributes,  
the algorithm calls for assigning the majority label as a prediction, but  
for very unbalanced datasets, the majority label will most likely be the most dominant class.  
Hence the algorithm will almost always assign the dominant class to all the leaf nodes  
once we run out of attributes.  

