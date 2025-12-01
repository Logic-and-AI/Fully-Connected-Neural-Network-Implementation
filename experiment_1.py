import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from fully_connected_nn import FullyConnectedNN


def generate_binary_problem(centers: np.ndarray, N: int = 100):
    """
    Generate a set of 2D points belonging in two classes

    N: int. Number of samples per class
    p: int. Number of dimensions
    centers: numpy.ndarray. A matrix whose columns correspond to the center
             of each class. Unit covariance matrix is assumed for all classes
    """

    rng = np.random.default_rng()
    # Class 0
    X0 = rng.multivariate_normal(centers[:, 0], np.eye(2), N)
    y0 = np.zeros(N)
    # Class 1
    X1 = rng.multivariate_normal(centers[:, 1], np.eye(2), N)
    y1 = np.ones(N)
    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1))

    return X, y

def generate_flower_problem():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y.ravel()


def plot_decision_boundary(model_method, X, y):
    """
    Plots the decision boundary for a model using a 2D dataset.

    Parameters:
    - model_method: function that takes (N, 2) input and returns (N,) predictions.
    - X: Input data of shape (N, 2).
    - y: Labels of shape (N,).
    """
    # Set min and max values for each feature and add some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01  # Step size for grid

    # Create a mesh grid covering the input space
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict values over the entire grid
    Z = model_method(np.c_[xx.ravel(), yy.ravel()])  # Pass reshaped (M, 2) input
    Z = Z.reshape(xx.shape)  # Reshape to match the grid shape

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

    # Plot the training examples
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title("Decision Boundary")
    plt.show()

def eval_accuracy(y_pred, y_true):
    accuracy = np.mean(y_pred == y_true)
    print("--------------------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("--------------------------------")


# ----------------------------- Πρώτο πείραμα: ----------------------------

#---------------------------
# generate binary, εκδοχή Α:
#---------------------------

centers=np.array([[0,8], [0,8]]).astype(int)
X, y=generate_binary_problem(centers, 1000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
input_size = 2
hidden_layers = [1, 3]
output_size = 1

model = FullyConnectedNN(input_size=input_size, hidden_size=hidden_layers, output_size=output_size, lr=1e-2)

y_pred_test = model.predict(X_test.T).flatten()
eval_accuracy(y_pred_test, y_test)

# training:
model.fit(X_train, y_train.reshape(-1, 1), iterations=5000, batch_size=None, show_step=500)

# predicting:
y_pred_test = model.predict(X_test.T).flatten()

eval_accuracy(y_pred_test, y_test)

model.plot_training_loss()
print('---------------------------------------')

# δεδομένου ότι η κλάση FullyConnected περιμένει το Χ.Τ :
def model_method(X_input):
    return model.predict(X_input.T).flatten()

plot_decision_boundary(model_method, X_test, y_test)


#---------------------------
# generate binary, εκδοχή Β:
#---------------------------

# ας δοκιμάσουμε να φέρουμε λίγο πιο κοντά τα κέντρα των κλάσεων:
centers_2=np.array([[3,5], [3,5]]).astype(int)
X_2, y_2=generate_binary_problem(centers_2, 1000)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_2, y_2, test_size=0.3, random_state=12)

# αφού τα δεδομένα είναι randomly generated και split, μια αύξηση της πολυπλοκότητας του μοντέλου δεν θα έφερνε απαραίτητα καλύτερο αποτέλεσμα σε test set accuracy
model_2 = FullyConnectedNN(input_size=input_size, hidden_size=hidden_layers, output_size=output_size, lr=1e-2)

y_pred_test_2 = model_2.predict(X_test2.T).flatten()
eval_accuracy(y_pred_test_2, y_test2)

# training (αφού είναι πιο δύσκολο το dataset, θα δοκιμάσουμε περισσότερα iterations):
model_2.fit(X_train2, y_train2.reshape(-1, 1), iterations=10000, batch_size=None, show_step=500)

# predicting:
y_pred_test_2 = model_2.predict(X_test2.T).flatten()

eval_accuracy(y_pred_test_2, y_test2)

model_2.plot_training_loss()
print('---------------------------------------')

def model_2_method(X_input):
    return model_2.predict(X_input.T).flatten()

plot_decision_boundary(model_2_method, X_test2, y_test2)


#---------------------------
# generate flower, εκδοχή A:
#---------------------------


X_3, y_3 = generate_flower_problem()
X_3 = X_3.T
y_3 = y_3.reshape(-1, 1)

X_train3, X_test3, y_train3, y_test3 = train_test_split(X_3, y_3, test_size=0.30, random_state=12)

scaler = StandardScaler()
X_train3 = scaler.fit_transform(X_train3)
X_test3 = scaler.transform(X_test3)

model_3 = FullyConnectedNN(input_size=2, hidden_size=[1, 3], output_size=1, lr=1e-2)

model_3.fit(X_train3, y_train3, iterations=10000, batch_size=None, show_step=1000)

y_pred_test_3 = model_3.predict(X_test3.T).flatten()

eval_accuracy(y_pred_test_3, y_test3.flatten())

model_3.plot_training_loss()
print('---------------------------------------')

def model_3_method(X_input):
    return model_3.predict(X_input.T).flatten()

plot_decision_boundary(model_3_method, X_test3, y_test3)


#---------------------------
# generate flower, εκδοχή B:
#---------------------------

# ας δοκιμάσουμε ένα μοντέλο με περισσότερα hidden layers, και, επιπλέον, να έχουμε το varying_lr ενεργοποιημένο:

model_4 = FullyConnectedNN(input_size=2, hidden_size=[3, 32], output_size=1, lr=3e-2, varying_lr=True, varying_lr_loss_horizon=10, varying_lr_coefficient=0.8)

model_4.fit(X_train3, y_train3, iterations=49000, batch_size=None, show_step=1000)

y_pred_test_4 = model_4.predict(X_test3.T).flatten()

eval_accuracy(y_pred_test_4, y_test3.flatten())

model_4.plot_training_loss()
print('---------------------------------------')

def model_4_method(X_input):
    return model_4.predict(X_input.T).flatten()

plot_decision_boundary(model_4_method, X_test3, y_test3)


#---------------------------
# generate flower, εκδοχή Γ:
#---------------------------

# ας δοκιμάσουμε χωρίς varying_lr, αλλά με μικρότερο lr και περισσότερα iterations στην εκπαίδευση:

model_5 = FullyConnectedNN(input_size=2, hidden_size=[3, 32], output_size=1, lr=5e-3, varying_lr=False)

model_5.fit(X_train3, y_train3, iterations=60000, batch_size=None, show_step=1000)

y_pred_test_5 = model_5.predict(X_test3.T).flatten()

eval_accuracy(y_pred_test_5, y_test3.flatten())

model_5.plot_training_loss()
print('---------------------------------------')

def model_5_method(X_input):
    return model_5.predict(X_input.T).flatten()

plot_decision_boundary(model_5_method, X_test3, y_test3)
