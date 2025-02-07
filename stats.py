import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# csv_file = "/Users/okpaleke34/Documents/Programming/businesses/SEM7/thesis/program/media/zf_processed_videos/c1_31/detected_humans_annoted.csv"
# csv_file = "/Users/okpaleke34/Documents/Programming/businesses/SEM7/thesis/program/media/zf_processed_videos/c1_38/detected_humans_annoted.csv" #scenario 2
csv_file = "/Users/okpaleke34/Documents/Programming/businesses/SEM7/thesis/program/media/zf_processed_videos/RefCam_FC/detected_humans_annoted.csv" #scenario 1
# Load the dataset
df = pd.read_csv(csv_file)  # Update path to your CSV


# Step 3: Performance (Speed) Analysis
# Calculate processing time statistics and visualize them.

# Ensure 'is_person' and 'has_face' are properly filled (1 or 0)
df = df[(df['is_person'] == 1) & (df['has_face'].notna())]
df['has_face'] = df['has_face'].astype(int)  # Ensure boolean values

# Models to analyze
models = ['mtcnn', 'retinaFace', 'yolov8', 'yolov11', 'opencv']
pt_columns = [f"{model}_pt" for model in models]

# Compute descriptive statistics
speed_stats = df[pt_columns].describe().transpose()[['mean', '50%', 'std', 'min', 'max']]
speed_stats.rename(columns={'50%': 'median'}, inplace=True)

# Display speed statistics table
print("Speed Statistics (in seconds):")
print(speed_stats)

# Plot average processing time
plt.figure(figsize=(10, 6))
sns.barplot(x='mean', y=speed_stats.index, data=speed_stats.reset_index(), palette='viridis')
plt.title('Average Processing Time by Model')
plt.xlabel('Time (seconds)')
plt.ylabel('Model')
plt.tight_layout()
plt.show()

# Plot processing time distributions
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[pt_columns], palette='viridis')
plt.xticks(rotation=45)
plt.title('Processing Time Distribution by Model')
plt.ylabel('Time (seconds)')
plt.tight_layout()
plt.show()




# Step 4: Accuracy Analysis
# Compare model results against human annotations (has_face).

# Calculate accuracy metrics
metrics = []
for model in models:
    res_col = f"{model}_res"
    if res_col not in df.columns:
        continue  # Skip models not in the dataset
    
    valid_rows = df[res_col].notna()
    y_true = df.loc[valid_rows, 'has_face']
    y_pred = df.loc[valid_rows, res_col].astype(int)  # Ensure boolean values
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics.append({
        'Model': model,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    })

metrics_df = pd.DataFrame(metrics).set_index('Model')

# # Display accuracy metrics table
# print("\nAccuracy Metrics:")
# print(metrics_df)

# Plot accuracy metrics
plt.figure(figsize=(12, 6))
metrics_df.plot(kind='bar', ax=plt.gca())
plt.title('Model Performance Metrics')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Step 5: Speed vs. Accuracy Trade-off
# Visualize the relationship between processing time and accuracy.

# Merge speed and accuracy data
combined_df = speed_stats[['mean']].merge(metrics_df[['Accuracy']], left_index=True, right_index=True)

# Plot trade-off
# plt.figure(figsize=(8, 6))
# sns.scatterplot(data=combined_df, x='mean', y='Accuracy', hue=combined_df.index, s=200)
# plt.title('Accuracy vs. Processing Time')
# plt.xlabel('Average Processing Time (s)')
# plt.ylabel('Accuracy')
# plt.grid(True)
# for model in combined_df.index:
#     plt.annotate(model, (combined_df.loc[model, 'mean'], combined_df.loc[model, 'Accuracy']), 
#                  textcoords="offset points", xytext=(0,10), ha='center')
# plt.tight_layout()
# plt.show()