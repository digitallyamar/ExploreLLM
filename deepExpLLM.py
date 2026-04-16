import torch
import torch.nn as nn

# 1. SETUP (Same as before)
chars = "APLE"
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

def get_vector(char):
    vec = torch.zeros(4)
    vec[char_to_int[char]] = 1.0
    return vec

# 2. THE DEEP MODEL
class DeepTinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: Input (4) -> Hidden Scratchpad (16 neurons)
        self.layer1 = nn.Linear(4, 16)
        # The "Filter": Only lets positive signals through
        self.relu = nn.ReLU()
        # Layer 2: Hidden (16) -> Output (4)
        self.layer2 = nn.Linear(16, 4)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

model = DeepTinyLLM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Faster turner

# 3. TRAINING (A->P, P->L, L->E)
facts = [
    (get_vector('A'), get_vector('P')),
    (get_vector('P'), get_vector('L')),
    (get_vector('L'), get_vector('E'))
]

print("Training Deep Model on 3 facts...")
for epoch in range(301):
    total_loss = 0
    for input_vec, target_vec in facts:
        optimizer.zero_grad()
        output = model(input_vec)
        loss = criterion(output, target_vec)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/3:.4f}")

# 4. TESTING
print("\n--- Deep Model Results ---")
for char in ['A', 'P', 'L']:
    with torch.no_grad():
        out = model(get_vector(char))
        pred = int_to_char[torch.argmax(out).item()]
        print(f"Input: {char} -> Predicted: {pred}")

# 5. THE WEIGHTS
print("\n--- Layer 1 Weights (4x16) ---")
print(model.layer1.weight.shape)
print("\n--- Layer 2 Weights (16x4) ---")
print(model.layer2.weight.shape)
