import torch
import torch.nn as nn

## This model is not useful in anyways as it is trained for only
## one input, where if input is character 'A' output will be 'P'.
## So at the end we can only test it on input 'A' predicting output 'P'.
## But it shows how LLMs store and retrieve information.

# 1. THE DATA: Mapping characters to numbers
chars = "APLE" 
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

# 2. THE MODEL (The "Equation")
class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 inputs (A,P,L,E) mapping to 4 outputs
        self.layer = nn.Linear(4, 4) 
    
    def forward(self, x):
        return self.layer(x)

model = TinyLLM()
criterion = nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) 

# 3. TRAINING DATA: Teaching 'A' -> 'P'
# 'A' is represented as [1, 0, 0, 0]
input_data = torch.tensor([1.0, 0.0, 0.0, 0.0]) 
# 'P' is represented as [0, 1, 0, 0]
target = torch.tensor([0.0, 1.0, 0.0, 0.0])     

print("Starting CPU Training...")
for epoch in range(101):
    output = model(input_data)
    loss = criterion(output, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 4. TEST THE RESULT
with torch.no_grad():
    prediction = model(input_data)
    predicted_index = torch.argmax(prediction).item()
    print(f"\nFinal Result -> Input: 'A' | Predicted: '{int_to_char[predicted_index]}'")
