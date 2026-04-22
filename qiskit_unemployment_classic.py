import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Same Set Up as Quantum
df = pd.read_csv("2020-01-01 to 2026-01-01 Unemployment Rate by Metropolitan Statistical Area (Percent).csv")
df = df.drop(columns=["Series ID", "Series Name", "Units", "Region Code"])

df_long = df.melt(
    id_vars=["Region Name"],
    var_name="Date",
    value_name="Unemployment"
)

df_long["Date"] = pd.to_datetime(df_long["Date"])
df_long["Unemployment"] = pd.to_numeric(df_long["Unemployment"], errors="coerce")

df_long = df_long.sort_values(["Region Name", "Date"])
df_long["Unemployment"] = df_long.groupby("Region Name")["Unemployment"].ffill()

df_long["Next"] = df_long.groupby("Region Name")["Unemployment"].shift(-1)
df_long["Label"] = (df_long["Next"] > df_long["Unemployment"]).astype(int)

# Sliding window logic from earlier
window_size = 4
x, y = [], []

for region in df_long["Region Name"].unique():
    sub = df_long[df_long["Region Name"] == region].sort_values("Date")
    values = sub["Unemployment"].values
    labels = sub["Label"].values

    for i in range(len(values) - window_size):
        x.append(values[i:i+window_size])
        y.append(labels[i+window_size-1])

x = np.array(x)
y = np.array(y)

# remove NaNs
valid = ~np.any(np.isnan(x), axis=1) & ~np.isnan(y)
x = x[valid][:200]
y = y[valid][:200]

# Now we normalize data and make testing splits
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# This is a simple regression model that comes with sklearn
model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Classical:")
print("Accuracy:", acc)