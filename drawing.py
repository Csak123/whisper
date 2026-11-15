import matplotlib.pyplot as plt
import json

# Read the JSON data from a file
with open('output/whisper-small-finetune/trainer_state.json', 'r') as file:
    data = json.load(file)

# Extract loss and eval_loss values along with their corresponding epochs
losses = [entry['loss'] for entry in data['log_history'] if 'loss' in entry]
eval_losses = [entry['eval_loss'] for entry in data['log_history'] if 'eval_loss' in entry]
epochs = [entry['epoch'] for entry in data['log_history'] if 'loss' in entry]
eval_epochs = [entry['epoch'] for entry in data['log_history'] if 'eval_loss' in entry]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, label='Training Loss')
plt.plot(eval_epochs, eval_losses, label='Evaluation Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
