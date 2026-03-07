import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to print messages slowly
def slow_print(message, delay=1):
    print(message)
    time.sleep(delay)

# Load preprocessed dataset
data = pd.read_csv("C:\\Users\\PMYLS\\Downloads\\jupiter_moons_dataset_aic.csv")
data.columns = data.columns.str.strip().str.replace('Â', '')  # Clean column names

# ------------------- Resource Availability Plot ------------------- #
slow_print("\nResource availability predictions:")
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Name',
    y='Predicted Resource Availability (tons/km²)',
    data=data.sort_values('Predicted Resource Availability (tons/km²)', ascending=False),
    palette='viridis'
)
plt.xticks(rotation=90)
plt.xlabel("Moon Name")
plt.ylabel("Resources Availability in Tons")
plt.title("Resource Availability Prediction")
plt.show()

# ------------------- ARIMA Forecast Plot ------------------- #
slow_print("\nARIMA Forecasted Sustainable Extraction Years:")
plt.figure(figsize=(12, 6))
sns.barplot(
    x='Name',
    y='ARIMA Forecasted Years',
    data=data.sort_values('ARIMA Forecasted Years', ascending=False),
    palette='viridis'
)
plt.xticks(rotation=90)
plt.xlabel("Moon Name")
plt.ylabel("Years of Sustainable Extraction")
plt.title("Sustainable Extraction Forecast for Jupiter's Moons")
plt.show()

# ------------------- Water Presence Pie Chart ------------------- #
water_presence_counts = data['Water presence (Y/N)'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(
    water_presence_counts,
    labels=['No Water (0)', 'Water Present (1)'],
    autopct='%1.1f%%',
    startangle=90,
    colors=['#FF6347', '#32CD32'],
    wedgeprops={'edgecolor': 'black'}
)
plt.title('Distribution of Water Presence on Jupiter Moons')
plt.axis('equal')
plt.show()

# ------------------- Water Presence by Moon Groups ------------------- #
water_group_counts = data.groupby(['Group', 'Water presence (Y/N)']).size().unstack(fill_value=0)
plt.figure(figsize=(12, 6))
ax = water_group_counts.plot(kind='bar', stacked=True, color=['#1E90FF', '#FFA07A'], figsize=(12, 6))
plt.xticks(ticks=range(len(water_group_counts.index)), labels=range(len(water_group_counts.index)), rotation=0, fontsize=10)
plt.title('Water Presence by Moon Groups')
plt.xlabel('Moon Group Index')
plt.ylabel('Count')
plt.legend(title='Water Presence', labels=['Water Present (1)', 'No Water (0)'])
plt.tight_layout()
plt.show()

# ------------------- Extraterrestrial Life Potential Pie Chart ------------------- #
extraterrestrial_counts = data['Extraterrestrial Life Potential'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(
    extraterrestrial_counts,
    labels=['No Life Potential (0)', 'Life Potential (1)'],
    autopct='%1.1f%%',
    startangle=90,
    colors=['#FF6347', '#32CD32'],
    wedgeprops={'edgecolor': 'black'}
)
plt.title('Distribution of Extraterrestrial Life Potential on Jupiter Moons')
plt.axis('equal')
plt.show()

# ------------------- Top and Bottom Moons by Predicted Life Probability ------------------- #
top_moons = data.sort_values('Predicted Life Probability', ascending=False).head(5)
bottom_moons = data.sort_values('Predicted Life Probability', ascending=True).head(5)

min_visible_prob = 0.03
bottom_moons['Adjusted Probability'] = bottom_moons['Predicted Life Probability'].clip(lower=min_visible_prob)
top_moons['Adjusted Probability'] = top_moons['Predicted Life Probability']

top_bottom_moons = pd.concat([top_moons, bottom_moons])
plt.figure(figsize=(12, 6))
sns.barplot(x='Name', y='Adjusted Probability', data=top_bottom_moons, palette='coolwarm')
plt.xticks(rotation=45)
plt.xlabel("Moon Name")
plt.ylabel("Predicted Life Probability")
plt.title("Moons with Highest and Lowest Probability of Extraterrestrial Life")
plt.show()

# ------------------- Estimated Life Longevity Plot ------------------- #
slow_print("Estimated Life Longevity of Jupiter Moons")
plt.figure(figsize=(15, 8))
sorted_data = data.sort_values(by='Estimated Life Longevity (years)', ascending=False)
sns.barplot(data=sorted_data, x='Name', y='Estimated Life Longevity (years)', palette='viridis')
plt.xticks(rotation=90, fontsize=8)
plt.xlabel('Moon Name')
plt.ylabel('Estimated Life Longevity (years)')
plt.title('Estimated Life Longevity of Jupiter Moons')
plt.show()

# ------------------- Extraterrestrial Life Potential by Moon Groups ------------------- #
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Group',
    y='Extraterrestrial Life Potential',
    data=data,
    errorbar=None,
    palette='coolwarm'
)
plt.title('Average Extraterrestrial Life Potential by Moon Groups')
plt.xlabel('Moon Group')
plt.ylabel('Average Extraterrestrial Life Potential')
plt.show()

# ------------------- Surface Composition Distribution ------------------- #
surface_composition_cols = [col for col in data.columns if 'Surface composition_' in col]
surface_composition_counts = data[surface_composition_cols].sum().sort_values(ascending=False)

bright_colors = ['#33FF62', '#33FF57', '#3357FF', '#FF33A1', '#33FF57', '#DAF7A6', '#581845', '#FF33A1']
plt.figure(figsize=(12, 6))
surface_composition_counts.plot(kind='bar', color=bright_colors[:len(surface_composition_counts)])
plt.title('Distribution of Surface Composition on Jupiter Moons')
plt.xlabel('Surface Composition')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# ------------------- Habitability Index vs Resource Availability Bubble Chart ------------------- #
unique_moons = data["Name"].unique()
colors = sns.color_palette("tab10", n_colors=len(unique_moons))
color_map = {moon: colors[i] for i, moon in enumerate(unique_moons)}

plt.figure(figsize=(12, 6))
bubble_size = data['Predicted Resource Availability (tons/km²)'] * 10

for moon in unique_moons:
    moon_data = data[data["Name"] == moon]
    plt.scatter(
        moon_data['Habitability Index'],
        moon_data['Predicted Resource Availability (tons/km²)'],
        s=bubble_size[moon_data.index],
        color=color_map[moon],
        label=moon, alpha=0.8, edgecolors="black"
    )
    for i in moon_data.index:
        plt.text(
            moon_data.loc[i, 'Habitability Index'],
            moon_data.loc[i, 'Predicted Resource Availability (tons/km²)'],
            moon_data.loc[i, 'Name'], fontsize=10, ha='center', va='center'
        )

plt.xlabel('Habitability Index')
plt.ylabel('Predicted Resource Availability (tons/km²)')
plt.title('Bubble Chart: Habitability Index vs Resource Availability for Selected Moons')
legend_patches = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[moon], markersize=10, label=moon)
    for moon in unique_moons
]
plt.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.3, 1))
plt.grid(True)
plt.show()
