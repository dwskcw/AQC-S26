import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_aer import AerSimulator
from qiskit import transpile
from scipy.optimize import minimize

# Basically I downloaded code from the FRED website and just made a script that 
# processes it for quantum computing usage => Dathan

# Load the dataset into dataframe
df = pd.read_csv("2020-01-01 to 2026-01-01 Unemployment Rate by Metropolitan Statistical Area (Percent).csv")

# We only want the region name and the unemployment rate for each date, so we drop the other columns
df = df.drop(columns=["Series ID", "Series Name", "Units", "Region Code"])

# This converts our table so we it turns our columns into rows, this makes it easier to plot and analyze the data
df_long = df.melt(
    id_vars=["Region Name"],
    var_name="Date",
    value_name="Unemployment"
)

# We need to convert our data from strings to the correct type
df_long["Date"] = pd.to_datetime(df_long["Date"]) 
df_long["Unemployment"] = pd.to_numeric(df_long["Unemployment"], errors="coerce")

# Error checking to guarantee that any null or missing data gets filled with last known value
df_long = df_long.sort_values(["Region Name", "Date"])  # We sort so that when we go in order or data we can fill in the data according to time
df_long["Unemployment"] = df_long.groupby("Region Name")["Unemployment"].ffill() # ffill -> forward fill, basically what was explained earlier

# The Idea here is to get the next value of unemployment and have it be side by side with the current
# value, this way we can label our data for training: 1 if the next value has higher unemployment,0 if
# it has lower unemployment
df_long["Next"] = df_long.groupby("Region Name")["Unemployment"].shift(-1)
df_long["Label"] = (df_long["Next"] > df_long["Unemployment"]).astype(int)  # this just inserts 0 or 1 respectively


window_size = 4

x = []
y = []

# Go through each region and with sorted data, basically go through and 
# creating training data by taking the unemployment values for every 4 months and then
# labeling with the next unemployment value. This makes it easy for our quantum model since
# we can have it learn patterns from the past four months of data to predict.
for region in df_long["Region Name"].unique():
    sub = df_long[df_long["Region Name"] == region].sort_values("Date") 
    values = sub["Unemployment"].values
    labels = sub["Label"].values

    #This is where we make each window of data
    for i in range(len(values) - window_size):
        
        # add windows and labels to training data
        x.append(values[i:i+window_size])
        y.append(labels[i+window_size-1])

# qiskit works with numpy arrays so convert
x = np.array(x)
y = np.array(y)

# Remove any rows with NaN values
valid_indices = ~np.any(np.isnan(x), axis=1) & ~np.isnan(y)
x = x[valid_indices]
y = y[valid_indices]

x = x[:200]
y = y[:200]

#print(f"Valid samples after removing NaN: {len(x)}")
# Switched to 200 samples because it was WAY TOO slow with 20000+ samples

# =====================================================================

# Quantum Data Preprocessing

# Since we got our data, we now need to normalize it between 0 and pi


scaler = MinMaxScaler((0, np.pi))
x = scaler.fit_transform(x)

# this is just doing a splitting of out data so we do 80% for training and 20% for testing 
# random_state=42 means we have 42 as a seed for a random number generator, to help us get an even split of data 
# (this could be any number)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# since we have 4 features (4 months of data for each window) we use 4 qubits, one for each feature
num_qubits = 4

# initialize feature map and ansatz (reps are how many times we repeat circuit)
feature_map = zz_feature_map(num_qubits, reps=2) # convert data to quantum state
ansatz = real_amplitudes(num_qubits, reps=6)     # circuit to do the learning

# example quantum simulator
sim = AerSimulator()
loss_history = []

# helper function from online
def parity(bitstring):
    return bitstring.count("1") % 2

# prediction ml function
def predict(params, x_sample):

    # this just makes the circuit from our current parameters and sample
    fm = feature_map.assign_parameters(dict(zip(feature_map.parameters, x_sample))) 
    an = ansatz.assign_parameters(params)

    # combine feature map and ansatz and measure qubits
    qc = fm.compose(an)
    qc.measure_all()

    # run the circuit on the simulator and get counts
    result = sim.run(qc, shots=256).result()
    counts = result.get_counts()

    # since this is binary classification, we just count the number of 1s (times we predict if higher upemployment)
    # We then take an average to see the probability of predicting higher unemployment
    total = 0
    for bitstring, count in counts.items():
        total += parity(bitstring) * count

    return total / 256 # we took 256 shots or runs of the circuit, so we divide by 256 to get the average prediction per sample

# loss function to figure out how bad our predictions are
def loss(params):
    # go through each sample
    preds = np.array([predict(params, x) for x in X_train])
    
    # we use this to basically make sure when we do the log in our calculation we don't get NaN
    temp = 1e-8
    preds = np.clip(preds, temp, 1 - temp)
    
    # this is a binary cross entropy function => common measurement for how far we predicted was
    err = -np.mean(y_train * np.log(preds) + (1 - y_train) * np.log(1 - preds))

    loss_history.append(err) 
    
    return err

# Training section 

# We give random number of parameters to start with
init = np.random.random(len(ansatz.parameters))

# using cobyla method we try to minimize loss within 50 iterations
result = minimize(
    loss,
    init,
    method="COBYLA",
    options={"maxiter": 50}
)
best_params = result.x


# once we finished training, we test our model
# we check for > 0.5 because this means that our model is confident
# that next month will have higher unemployment and if so we predict 1, 
# otherwise we predict 0, then we basically make a plot that is just a true or false
# of if we were correct. We then take the average of that to get our accuracy.
test_preds = np.array([predict(best_params, x) > 0.5 for x in X_test])
acc = np.mean(test_preds == y_test) 

# this is just matplotlib graphing ==> the first graph is loss history
# the second generated graph is just how accurate we were compared to randomly guessing
# we prob only really need the print statement **
print("Quantum:")
print("Accuracy:", acc)
plt.figure()
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()