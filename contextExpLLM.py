import torch
import torch.nn as nn

## VERSION 4: The "Context" LLM
## Enhancement: Uses a Hidden Layer AND multi-character context.
## By looking at TWO letters, the model knows if it's at the start or middle.

chars = "APLE"
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

# We now look at 2 characters at once (4 bits + 4 bits = 8 inputs)
def encode_context(two_chars):
    v1 = torch.zeros(4)
    v1[char_to_int[two_chars[0]]] = 1.0
    v2 = torch.zeros(4)
    v2[char_to_int[two_chars[1]]] = 1.0
    return torch.cat((v1, v2)) # Combine into one 8-length vector

class DeepTinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        # 8 inputs -> 12 hidden neurons -> 4 outputs
        self.layer1 = nn.Linear(8, 12)
        self.relu = nn.ReLU() # The "Thinking" step
        self.layer2 = nn.Linear(12, 4)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        return self.layer2(x)

model = DeepTinyLLM()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# TRAINING DATA (Sliding Window)
# "AP" -> 'P'
# "PP" -> 'L'
# "PL" -> 'E'
inputs = torch.stack([
    encode_context("AP"),
    encode_context("PP"),
    encode_context("PL")
])

targets = torch.stack([
    torch.tensor([0., 1., 0., 0.]), # P
    torch.tensor([0., 0., 1., 0.]), # L
    torch.tensor([0., 0., 0., 1.])  # E
])

print("Starting V4 Training (Deep Learning + Context)...")
for epoch in range(501):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# TEST
print("\n--- Context Spelling Test ---")
with torch.no_grad():
    pairs = ["AP", "PP", "PL"]
    for p in pairs:
        pred = model(encode_context(p))
        char = int_to_char[torch.argmax(pred).item()]
        print(f"Context: '{p}' -> Predicted Next: '{char}'")
