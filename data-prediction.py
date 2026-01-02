import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
df = pd.read_csv("powerplant_data_edited.csv")

print(df.describe())

corr_matrix = df.corr(method='pearson')

# Extract correlation with energy_output
energy_corr = corr_matrix['energy_output'].sort_values(ascending=False)

print("pearson correlation")
print(energy_corr)



X = df.drop('energy_output', axis=1)
y = df['energy_output']
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y)

rf_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print()
print('###############################')
print("Random forest correlation")
print(rf_importance)


df.plot(
    kind='scatter',
    x='temperature',
    y='energy_output',
    title='Temperature vs Energy Output',
    grid=True
)

plt.figure()
plt.scatter(df['temperature'], df['energy_output'])
plt.xlabel('Temperature')
plt.ylabel('Energy Output')
plt.title('Temperature vs Energy Output')
plt.grid(True)
plt.show()