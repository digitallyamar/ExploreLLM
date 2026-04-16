import torch
import torch.nn as nn

## VERSION 3: The "Sequence" LLM
## Enhancement: Teaching the model to predict the next letter in "APPLE".
## Note: Because 'P' is followed by both 'P' and 'L', the model will 
## likely "average" the two and pick the strongest one.

# 1. THE DATA
chars = "APLE" 
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

# Helper to turn a string into "One-Hot" tensors
def encode(text):
    base = torch.zeros(len(text), 4)
    for i, ch in enumerate(text):
        base[i, char_to_int[ch]] = 1.0
    return base

# 2. THE MODEL (Still our reliable 16-parameter brain)
class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4) 
    
    def forward(self, x):
        return self.layer(x)

model = TinyLLM()
criterion = nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) 

# 3. SEQUENCE DATA
# Inputs:  A  P  P  L
# Targets: P  P  L  E
input_seq = encode("APPL")
target_seq = encode("PPLE")

print("Starting V3 Training (Sequence Learning)...")
for epoch in range(201): # More epochs to handle the conflict
    outputs = model(input_seq)
    loss = criterion(outputs, target_seq)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 40 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 4. TESTING THE SEQUENCE
print("\n--- Spelling Test ---")
with torch.no_grad():
    test_chars = "APPL"
    preds = model(encode(test_chars))
    indices = torch.argmax(preds, dim=1)
    
    result = "".join([int_to_char[idx.item()] for idx in indices])
    
    for i, char in enumerate(test_chars):
        print(f"Input: '{char}' -> Predicted Next: '{result[i]}'")

print(f"\nFinal Sequence Result: {test_chars[0]} -> {result}")
