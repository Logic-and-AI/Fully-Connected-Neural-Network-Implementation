import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Χρήση συσκευής: {device}")

data_bc = load_breast_cancer()
X = data_bc.data
y = data_bc.target.reshape(-1, 1)

def eval_accuracy(y_pred, y_true):
    y_pred_classes = (y_pred > 0.5).float()
    return (y_pred_classes.flatten() == y_true.flatten()).float().mean().item()


class SimpleFCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleFCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


start_time = time.time()

# -------------------------------------------------
# Βασικές παράμετροι
# -------------------------------------------------
n_repeats = 20
batch_size = 64
total_batches = 5000
test_accuracies = []

for run in range(n_repeats):
    print(f'Επανάληψη {run+1} από {n_repeats}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=None
    )

    # Scaling
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Μετατροπή σε torch tensors και μεταφορά στη συσκευή
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Μοντέλο
    model = SimpleFCNN(input_dim=X.shape[1], hidden_dim=16, output_dim=1).to(device)

    # Optimizer και Loss
    loss_function = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    n_epochs=total_batches # απλά διασφαλίζουμε ότι έχουμε έναν αρκετά μεγάλο αριθμο εποχών διαθέσιμο, η εκπαίδευση σε κάθε περίπτωση θα διακοπεί πιο πριν
    iteration_counter=0

    # Training
    model.train()
    for epoch in range(n_epochs):
      for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        model_values=model(x_batch)
        loss=loss_function(model_values, y_batch)
        loss.backward()
        optimizer.step()
        iteration_counter+=1
        if iteration_counter % 500 == 0:
            print(f"Iteration: {iteration_counter}, Loss: {loss.item():.4f}")
        if iteration_counter >= total_batches:
            break
      if iteration_counter >= total_batches:
        break

    # Testing
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor)
        accuracy_result = eval_accuracy(y_pred_test, y_test_tensor)

    test_accuracies.append(accuracy_result)

end_time = time.time()
elapsed_time = end_time - start_time
mean_accuracy = np.mean(test_accuracies)
std_accuracy = np.std(test_accuracies)

print("\n==========================================")
print(f"Μέση ακρίβεια Test Set: {mean_accuracy*100:.2f}%")
print(f"Τυπική απόκλιση Test Set: {std_accuracy*100:.2f}%")
print(f"Συνολικός Χρόνος Εκτέλεσης: {elapsed_time:.2f} δευτερόλεπτα")
print("============================================")
