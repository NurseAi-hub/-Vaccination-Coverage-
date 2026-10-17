import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

file_name = 'vaccin-child.csv'
try:
    df = pd.read_csv(file_name)
    print("✅ Data successfully loaded!")
    print(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
except FileNotFoundError:
    print("❌ Error: File not found. Please upload 'vaccin-child.csv'.")


le_group = LabelEncoder()
df['group_encoded'] = le_group.fit_transform(df['Group'].astype(str))

le_cat = LabelEncoder()
# Pay attention: the 'Category' column name might contain a space, so we use the correct column variable
cat_column = 'Category ' if 'Category ' in df.columns else 'Category'
df['category_encoded'] = le_cat.fit_transform(df[cat_column].astype(str))

le_vac = LabelEncoder()
df['vaccination_encoded'] = le_vac.fit_transform(df['Vaccination'].astype(str))


df = df.dropna(subset=['Estimate', 'Year', 'category_encoded', 'group_encoded', 'vaccination_encoded'])


X = df[['category_encoded', 'Year', 'group_encoded', 'vaccination_encoded']].values
y = df['Estimate'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# . Build and train model Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# . results
y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\\n✅ Updated Root Mean Squared Error (RMSE): {rmse:.2f}%")
print(f"✅ Updated R-squared (R^2) Score: {r2:.2f}")


importances = rf_model.feature_importances_
features = ['Category', 'Year', 'Group', 'Vaccination']

plt.figure(figsize=(8, 4))
plt.barh(features, importances * 100, color='teal')
plt.xlabel('Importance (%)')
plt.title('Feature Importance for Vaccination Coverage')
plt.show()


