import torch
import torch.nn as nn

## VERSION 2: The "Multi-Input" LLM
## Enhancement: This version teaches the model two distinct rules:
## 1. If input is 'A', predict 'P'
## 2. If input is 'P', predict 'P'
## It uses "Batching" to process both rules at the same time.

# 1. THE DATA
chars = "APLE" 
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

# 2. THE MODEL (Stays the same - 16 parameters)
class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4) 
    
    def forward(self, x):
        return self.layer(x)

model = TinyLLM()
criterion = nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) 

# 3. BATCHED TRAINING DATA
# We stack 'A' and 'P' into one tensor (2 rows, 4 columns)
# Input Row 0: 'A' [1,0,0,0] | Input Row 1: 'P' [0,1,0,0]
inputs = torch.tensor([
    [1.0, 0.0, 0.0, 0.0], 
    [0.0, 1.0, 0.0, 0.0]
]) 

# Target Row 0: 'P' [0,1,0,0] | Target Row 1: 'P' [0,1,0,0]
targets = torch.tensor([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]
])     

print("Starting V2 Training (Batch Processing)...")
for epoch in range(101):
    # The model processes BOTH inputs in one shot here
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    optimizer.zero_grad() # Empty the "buckets"
    loss.backward()      # Fill the "buckets"
    optimizer.step()     # Update the weights
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 4. TEST BOTH INPUTS
print("\n--- Final Results ---")
with torch.no_grad():
    predictions = model(inputs)
    # Get the winning index for each row in the batch
    predicted_indices = torch.argmax(predictions, dim=1)
    
    for i, idx in enumerate(predicted_indices):
        input_char = "A" if i == 0 else "P"
        print(f"Input: '{input_char}' | Predicted: '{int_to_char[idx.item()]}'")

# 5. OPTIONAL: Save the "Brain"
# torch.save(model.state_dict(), 'v2_multi_input.pth')
