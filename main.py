
import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, recall_score, accuracy_score, \
    precision_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from statsmodels.tsa.arima.model import ARIMA
from dl_prediction import SimpleANN  # Importing the ANN model
from sklearn.model_selection import train_test_split


# Function to print messages slowly
def slow_print(message, delay=1):
    print(message)
    time.sleep(delay)


def select_features_for_resource_availability(data):
    # Select relevant features for predicting resource availability.

    selected_features = [ 'Extraction Rate (tons/year)', 'Replenishment Rate (tons/year)',
                          'Crust Stability Index', 'Operational Risk Factor' ]
    return data.dropna(subset=selected_features)


def apply_arima_forecasting(data):

    #Apply ARIMA forecasting on 'Years of Sustainable Extraction' column.

    data[ 'ARIMA Forecasted Years' ] = np.nan
    for index in range(len(data)):
        try:
            history = data.loc[ :index, 'Years of Sustainable Extraction' ].dropna().values[ -5: ]
            if len(history) < 5:
                data.at[ index, 'ARIMA Forecasted Years' ] = data.at[ index, 'Years of Sustainable Extraction' ]
            else:
                model = ARIMA(history, order=(1, 1, 0))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=1)[ 0 ]
                data.at[ index, 'ARIMA Forecasted Years' ] = max(forecast, 0)
        except Exception:
            data.at[ index, 'ARIMA Forecasted Years' ] = data.at[ index, 'Years of Sustainable Extraction' ]
    return data


def convert_temperature_range(temp_range):
    if 'to' in str(temp_range):
        lower, upper = map(int, temp_range.split(' to '))
        return (lower + upper) / 2
    return np.nan


def estimate_life_longevity(row):

    # Higher radiation levels and extreme temperatures lower the longevity estimate.

    if row[ 'Probability of Life Presence (%)' ] < 10:
        return 0
    elif row[ 'Radiation levels (mSv/year)' ] > 0.1:
        return np.random.randint(1, 50)
    elif row[ 'Temperature range (C)' ] < -150:
        return np.random.randint(10, 500)
    elif row[ 'Water presence (Y/N)' ] == 1:
        return np.random.randint(100, 10000)
    else:
        return np.random.randint(50, 1000)


# Load dataset
data = pd.read_csv("C:\\Users\\PMYLS\\Downloads\\jupiter_moons_dataset_aic.csv")
data.columns = data.columns.str.strip().str.replace('Â', '')  # Clean column names

# Handle missing values
data.replace([ np.inf, -np.inf ], np.nan, inplace=True)
data.fillna(data.median(numeric_only=True), inplace=True)

# Feature Selection for Resource Availability
data = select_features_for_resource_availability(data)

# Predicting Resource Availability using Random Forest
availability_features = [ 'Crust Stability Index', 'Operational Risk Factor', 'Replenishment Rate (tons/year)' ]
X_availability = data[ availability_features ]
y_availability = data[ 'Resource Availability (tons/km²)' ]

X_train_avail, X_test_avail, y_train_avail, y_test_avail = train_test_split(X_availability, y_availability,
                                                                            test_size=0.3, random_state=123)

slow_print("Training Random Forest Regressor for Resource Availability...")
avail_model = RandomForestRegressor(n_estimators=100, random_state=123)
avail_model.fit(X_train_avail, y_train_avail)

data[ 'Predicted Resource Availability (tons/km²)' ] = avail_model.predict(X_availability)

data[ 'Years of Sustainable Extraction' ] = data.apply(
    lambda row: row[ 'Predicted Resource Availability (tons/km²)' ] / max(
        row[ 'Extraction Rate (tons/year)' ] - row[ 'Replenishment Rate (tons/year)' ], 1), axis=1
)

# ARIMA Forecasting
slow_print("Applying ARIMA forecasting...")
warnings.filterwarnings("ignore", category=UserWarning)

data = apply_arima_forecasting(data)

slow_print("\nResource availability predictions:")
print(data[ [ 'Name', 'Predicted Resource Availability (tons/km²)' ] ])
# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x='Name', y='Predicted Resource Availability (tons/km²)',
            data=data.sort_values('Predicted Resource Availability (tons/km²)', ascending=False), palette='viridis')
plt.xticks(rotation=90)
plt.xlabel("Moon Name")
plt.ylabel("Resources Availability in Tons")
plt.title("Resource Availability Prediction")
plt.show()

slow_print("\nARIMA Forecasted Sustainable Extraction Years:")
print(data[ [ 'Name', 'ARIMA Forecasted Years' ] ])

# Visualization: Sustainable Extraction Forecast
plt.figure(figsize=(12, 6))
sns.barplot(x='Name', y='ARIMA Forecasted Years', data=data.sort_values('ARIMA Forecasted Years', ascending=False),
            palette='viridis')
plt.xticks(rotation=90)
plt.xlabel("Moon Name")
plt.ylabel("Years of Sustainable Extraction")
plt.title("Sustainable Extraction Forecast for Jupiter's Moons")
plt.show()

# Predicting Extraterrestrial Life Potential
data = data.dropna()
data[ 'Water presence (Y/N)' ] = data[ 'Water presence (Y/N)' ].map({'Y': 1, 'N': 0})
le = LabelEncoder()
data[ 'Group' ] = le.fit_transform(data[ 'Group' ])
data[ 'Potential biosignatures' ] = data[ 'Potential biosignatures' ].map({'Detected': 1, 'Not Detected': 0})

data[ 'Temperature range (C)' ] = data[ 'Temperature range (C)' ].apply(convert_temperature_range)
data = pd.get_dummies(data, columns=[ 'Surface composition', 'Atmosphere composition' ], drop_first=True)
data.fillna(data.median(numeric_only=True), inplace=True)

# Features for Life Probability Prediction
life_features = [ 'Water presence (Y/N)', 'Temperature range (C)', 'Radiation levels (mSv/year)',
                  'Potential biosignatures', 'Solar energy (W/m²)' ]
X_life = data[ life_features ]
y_life = np.where(
    (data[ 'Water presence (Y/N)' ] == 1) & (data[ 'Temperature range (C)' ] > -150) & (
            data[ 'Radiation levels (mSv/year)' ] < 0.1),
    1, 0
)
# Separate classes
X_majority = X_life[ y_life == 0 ]
X_minority = X_life[ y_life == 1 ]
y_majority = y_life[ y_life == 0 ]
y_minority = y_life[ y_life == 1 ]

# Split each separately
X_train_major, X_test_major, y_train_major, y_test_major = train_test_split(
    X_majority, y_majority, test_size=0.2, random_state=42
)
X_train_minor, X_test_minor, y_train_minor, y_test_minor = train_test_split(
    X_minority, y_minority, test_size=0.2, random_state=42
)

# Merge
X_train_life = np.vstack([ X_train_major, X_train_minor ])
y_train_life = np.hstack([ y_train_major, y_train_minor ])
X_test_life = np.vstack([ X_test_major, X_test_minor ])
y_test_life = np.hstack([ y_test_major, y_test_minor ])

# Replacing Random Forest Classifier with ANN Model
slow_print("Training ANN Model for Probability of Life Presence...")
# Initialize ANN model
ann_model = SimpleANN(input_size=X_train_life.shape[ 1 ])

# Train ANN model
ann_model.fit(X_train_life, y_train_life, epochs=100, lr=0.01)

# Make predictions
data[ 'Probability of Life Presence (%)' ] = ann_model.forward(X_life) * 100

# Estimating Life Longevity Based on Environmental Conditions

data[ 'Estimated Life Longevity (years)' ] = data.apply(estimate_life_longevity, axis=1)

# Display Results
print("\nProbability of Life Presence and Estimated Longevity:")
print(data[ [ 'Name', 'Estimated Life Longevity (years)' ] ])
print(data[ [ 'Name', 'Probability of Life Presence (%)' ] ])

data[ 'Extraterrestrial Life Potential' ] = np.where(
    (data[ 'Water presence (Y/N)' ] == 1) & (data[ 'Temperature range (C)' ] > -150) & (
            data[ 'Radiation levels (mSv/year)' ] < 0.1), 1, 0)

X = data.drop(columns=[ 'Name', 'Water presence (Y/N)', 'Extraterrestrial Life Potential' ])
y = data[ 'Extraterrestrial Life Potential' ]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=49)

print("\nEvaluating Random Forest Regressor (Resource Availability)...")
y_pred_avail = avail_model.predict(X_test_avail)
print(f"MAE: {mean_absolute_error(y_test_avail, y_pred_avail):.4f}")
print(f"MSE: {mean_squared_error(y_test_avail, y_pred_avail):.4f}")
print(f"R² Score: {r2_score(y_test_avail, y_pred_avail):.4f}")

print("\nEvaluating ARIMA Forecast (Sustainable Extraction Years)...")
arima_actual = data[ 'Years of Sustainable Extraction' ].dropna()
arima_predicted = data[ 'ARIMA Forecasted Years' ].dropna()

if len(arima_actual) == len(arima_predicted):  # Ensure lengths match
    print(f"MAE: {mean_absolute_error(arima_actual, arima_predicted):.4f}")
    print(f"MSE: {mean_squared_error(arima_actual, arima_predicted):.4f}")
    print(f"R² Score: {r2_score(arima_actual, arima_predicted):.4f}")

# Evaluating the Artificial Neural Network (ANN)
threshold = 0.3  # Experiment with different values
ann_predictions = (ann_model.predict(X_test_life) > threshold).astype(int)
ann_predictions = (ann_predictions > 0.5).astype(int)

# Convert probabilities to binary classification
print("Class Distribution in y_test_life:", np.bincount(y_test_life.flatten()))
# ANN Performance Metrics
print("\nArtificial Neural Network (ANN) Performance:")
print(f"Accuracy: {accuracy_score(y_test_life, ann_predictions):.4f}")

# Full classification report (includes per-class metrics)
print("\nDetailed Classification Report:\n")
print(classification_report(y_test_life, ann_predictions))

slow_print("Training classification models...")
logistic_model = LogisticRegression()
random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=78, random_state=42)

logistic_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

# Model Evaluation
models = {
    'Logistic Regression': logistic_model,
    'Random Forest Classifier': random_forest_model,

}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n{name} Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

print("\nBest Classification Model based on Accuracy:")
best_model = max(models, key=lambda m: accuracy_score(y_test, models[ m ].predict(X_test)))
slow_print(f"Best Classification Model: {best_model}")

slow_print("\nSummary of Model Performances:")
print("- Random Forest Regressor: Resource Availability Prediction")
print("- ARIMA: Sustainable Extraction Forecasting")
print("- Logistic Regression & Random Forest: Life Potential Classification")
print("- ANN: Probability of Life Presence Prediction")
# Visualize the distribution of water presence among Jupiter's moons

water_presence_counts = data[ 'Water presence (Y/N)' ].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 6))
plt.pie(water_presence_counts,
        labels=[ 'No Water (0)', 'Water Present (1)' ],
        autopct='%1.1f%%',
        startangle=90,
        colors=[ '#FF6347', '#32CD32' ],
        wedgeprops={'edgecolor': 'black'})

plt.title('Distribution of Water Presence on Jupiter Moons')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
plt.show()

# Visualization: Water Presence by Moon Groups

water_group_counts = data.groupby([ 'Group', 'Water presence (Y/N)' ]).size().unstack(fill_value=0)

# Plot the stacked bar chart with better colors
plt.figure(figsize=(12, 6))
ax = water_group_counts.plot(kind='bar', stacked=True, color=[ '#1E90FF', '#FFA07A' ], figsize=(12, 6))

# Use numerical indexes on x-axis
plt.xticks(ticks=range(len(water_group_counts.index)), labels=range(len(water_group_counts.index)), rotation=0,
           fontsize=10)

# Customize labels and title
plt.title('Water Presence by Moon Groups')
plt.xlabel('Moon Group Index')
plt.ylabel('Count')
plt.legend(title='Water Presence', labels=[ 'Water Present (1)', 'No Water (0)' ])

plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()

slow_print("Generating results for Extraterrestrial life potential")

# Visualize the probability of extraterrestrial life on Jupiter's moons using a pie chart

extraterrestrial_counts = data[ 'Extraterrestrial Life Potential' ].value_counts()
# Plotting the pie chart
plt.figure(figsize=(8, 6))
plt.pie(extraterrestrial_counts,
        labels=[ 'No Life Potential (0)', 'Life Potential (1)' ],
        autopct='%1.1f%%',
        startangle=90,
        colors=[ '#FF6347', '#32CD32' ],
        wedgeprops={'edgecolor': 'black'})

plt.title('Distribution of Extraterrestrial Life Potential on Jupiter Moons')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
plt.show()

# predicting Moons with the highest probability of extraterrestrial life
data[ 'Predicted Life Probability' ] = models[ best_model ].predict_proba(X_scaled)[ :, 1 ]

top_moons = data.sort_values('Predicted Life Probability', ascending=False).head(5)[
    [ 'Name', 'Predicted Life Probability' ] ]
print("\nMoons with the highest probability of extraterrestrial life:")
print(top_moons)

slow_print("Estimated Life Longevity of Jupiter Moons")
# Visualization : Estimated Life Longevity (years)
plt.figure(figsize=(15, 8))
sorted_data = data.sort_values(by='Estimated Life Longevity (years)', ascending=False)
sns.barplot(data=sorted_data, x='Name', y='Estimated Life Longevity (years)', palette='viridis')
plt.xticks(rotation=90, fontsize=8)
plt.xlabel('Moon Name')
plt.ylabel('Estimated Life Longevity (years)')
plt.title('Estimated Life Longevity of Jupiter Moons')
plt.show()

# Visualization: Extraterrestrial Life Potential by Moon Groups
plt.figure(figsize=(10, 6))
sns.barplot(x='Group', y='Extraterrestrial Life Potential', data=data, errorbar=None, palette='coolwarm')
plt.title('Average Extraterrestrial Life Potential by Moon Groups')
plt.xlabel('Moon Group')
plt.ylabel('Average Extraterrestrial Life Potential')
plt.show()

# Identifying Moons with the lowest probability of extraterrestrial life
bottom_moons = data.sort_values('Predicted Life Probability', ascending=True).head(5)[
    [ 'Name', 'Predicted Life Probability' ] ]

# Adjust probabilities to ensure visibility
min_visible_prob = 0.03  # Set a small baseline probability for visibility
bottom_moons[ 'Adjusted Probability' ] = bottom_moons[ 'Predicted Life Probability' ].clip(lower=min_visible_prob)
top_moons[ 'Adjusted Probability' ] = top_moons[ 'Predicted Life Probability' ]  # Keep original values for top moons

# Combine both datasets
top_bottom_moons = pd.concat([ top_moons, bottom_moons ])
slow_print("Moons with Highest and Lowest Probability of Extraterrestrial Life")
# Visualization with adjusted probabilities
plt.figure(figsize=(12, 6))
sns.barplot(x='Name', y='Adjusted Probability', data=top_bottom_moons, palette='coolwarm')
plt.xticks(rotation=45)
plt.xlabel("Moon Name")
plt.ylabel("Predicted Life Probability")
plt.title("Moons with Highest and Lowest Probability of Extraterrestrial Life :")
plt.show()

# Visualization: Surface Composition Distribution
surface_composition_cols = [ col for col in data.columns if 'Surface composition_' in col ]
surface_composition_counts = data[ surface_composition_cols ].sum().sort_values(ascending=False)

# Define a bright color palette
bright_colors = [ '#33FF62', '#33FF57', '#3357FF', '#FF33A1', '#33FF57', '#DAF7A6', '#581845', '#FF33A1' ]

plt.figure(figsize=(12, 6))
surface_composition_counts.plot(kind='bar', color=bright_colors[ :len(surface_composition_counts) ])
plt.title('Distribution of Surface Composition on Jupiter Moons')
plt.xlabel('Surface Composition')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
# Compute Habitability Index (weighted sum of key factors)
data[ 'Habitability Index' ] = (
                                       data[ 'Water presence (Y/N)' ] * 0.4 +
                                       (1 - (abs(data[
                                                     'Temperature range (C)' ] - 15) / 300)) * 0.3 +  # Normalized temp
                                       (1 - data[ 'Radiation levels (mSv/year)' ] / max(
                                           data[ 'Radiation levels (mSv/year)' ])) * 0.3  # Inverse radiation impact
                               ) * 100  # Scale to 0-100

# Ranking moons based on Habitability Index
habitability_ranking = data[ [ 'Name', 'Habitability Index' ] ].sort_values(by='Habitability Index', ascending=False)

# Print ranked table
slow_print("\nMoons Ranked by Habitability Index:")
print(habitability_ranking)

# Select 5 moons with the highest and 5 with the lowest Habitability Index
top_moons = data.nlargest(5, 'Habitability Index')
bottom_moons = data.nsmallest(5, 'Habitability Index')
selected_moons = pd.concat([ top_moons, bottom_moons ])

# Generate unique colors for these 10 moons
unique_moons = selected_moons[ "Name" ].unique()
colors = sns.color_palette("tab10", n_colors=len(unique_moons))
color_map = {moon: colors[ i ] for i, moon in enumerate(unique_moons)}

plt.figure(figsize=(12, 6))

# Normalize bubble sizes for better visualization
bubble_size = selected_moons[ 'Predicted Resource Availability (tons/km²)' ] * 10

# Plot each moon separately with a unique color
for moon in unique_moons:
    moon_data = selected_moons[ selected_moons[ "Name" ] == moon ]
    plt.scatter(moon_data[ 'Habitability Index' ],
                moon_data[ 'Predicted Resource Availability (tons/km²)' ],
                s=bubble_size[ moon_data.index ],
                color=color_map[ moon ],
                label=moon, alpha=0.8, edgecolors="black")

    # Adding the moon's name as text in the graph
    for i in moon_data.index:
        plt.text(moon_data.loc[ i, 'Habitability Index' ],
                 moon_data.loc[ i, 'Predicted Resource Availability (tons/km²)' ],
                 moon_data.loc[ i, 'Name' ], fontsize=10, ha='center', va='center')
slow_print("\nHabitability Index vs Resource Availability ")
# Labels and Title
plt.xlabel('Habitability Index')
plt.ylabel('Predicted Resource Availability (tons/km²)')
plt.title('Bubble Chart: Habitability Index vs Resource Availability for Selected Moons')

# Add Legend to indicate moon colors
legend_patches = [
    plt.Line2D([ 0 ], [ 0 ], marker='o', color='w', markerfacecolor=color_map[ moon ], markersize=10, label=moon)
    for moon in unique_moons ]
plt.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.3, 1))

plt.grid(True)
plt.show()

