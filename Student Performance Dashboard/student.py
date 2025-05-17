from IPython.display import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Reload the dataset
df = pd.read_csv("./Student Performance Dashboard/student_data.csv")

# 1. Top 5 Students by Average Marks
top_students = df.groupby("Name")["Marks"].mean(
).sort_values(ascending=False).head(5)

# 2. Top Scorer in Each Subject
top_subjects = df.loc[df.groupby("Subject")["Marks"].idxmax()]

# 3. Performance Categories
avg_marks = df.groupby("Name")["Marks"].mean()
conditions = [
    (avg_marks >= 85),
    (avg_marks >= 70),
    (avg_marks >= 50),
    (avg_marks < 50)
]
categories = ['Excellent', 'Good', 'Average', 'Needs Improvement']
performance_category = pd.Series(
    np.select(conditions, categories), index=avg_marks.index)

# 4. Grade Assignment
conditions_grade = [
    (df["Marks"] >= 90),
    (df["Marks"] >= 75) & (df["Marks"] < 90),
    (df["Marks"] >= 60) & (df["Marks"] < 75),
    (df["Marks"] >= 40) & (df["Marks"] < 60),
    (df["Marks"] < 40)
]
grades = ['A', 'B', 'C', 'D', 'F']
df["Grade"] = np.select(conditions_grade, grades)

# 5. Grade Distribution Pie Chart
grade_counts = df["Grade"].value_counts()
plt.figure(figsize=(6, 6))
grade_counts.plot(kind='pie', autopct='%1.1f%%', title="Grade Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig("./Student Performance Dashboard/grade_distribution.png")
plt.show()
plt.close()

# 6. Subject-wise Average Marks Bar Chart
subject_avg = df.groupby("Subject")["Marks"].mean()
plt.figure(figsize=(8, 5))
subject_avg.plot(kind='bar', color='skyblue', title="Average Marks by Subject")
plt.ylabel("Average Marks")
plt.xlabel("Subject")
plt.tight_layout()
plt.savefig("./Student Performance Dashboard/subject_average.png")
plt.show()
plt.close()

# 7. Heatmap for Subject Difficulty
pivot = df.pivot_table(index="Name", columns="Subject", values="Marks")
plt.figure(figsize=(12, 10))
sns.heatmap(pivot, cmap="coolwarm", cbar_kws={
            'label': 'Marks'}, linewidths=0.5)
plt.title("Student Marks Heatmap")
plt.tight_layout()
plt.savefig("./Student Performance Dashboard/marks_heatmap.png")
plt.show()
plt.close()

# Save performance categories to CSV
performance_summary = avg_marks.to_frame(name="Average Marks")
performance_summary["Category"] = performance_category
performance_summary_path = "./Student Performance Dashboard/student_performance_summary.csv"
performance_summary.to_csv(performance_summary_path)

# Return file paths to images and CSV
{
    "Grade Distribution Chart": "./Student Performance Dashboard/grade_distribution.png",
    "Subject Average Chart": "./Student Performance Dashboard/subject_average.png",
    "Heatmap": "./Student Performance Dashboard/marks_heatmap.png",
    "Performance Summary CSV": performance_summary_path
}
