import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(f"Initial Row Count: {len(df)}")

print(df.head())
print(df.info())
print(df.describe())

# (Clean Text)
df['Sex'] = df['Sex'].str.lower().str.strip()
df['Embarked'] = df['Embarked'].str.lower().str.strip()

#  (Missing Values)
df['Age'] = df['Age'].fillna(df['Age'].mean()).astype(int)
df['Cabin'] = df['Cabin'].fillna("unknown")
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

#  ID (Remove Duplicates)
df.drop_duplicates(subset=['PassengerId'], inplace=True)

# 4. Encoding
df['Sex_Code'] = df['Sex'].map({'male': 1, 'female': 0})


le = LabelEncoder()
df['Embarked_Code'] = le.fit_transform(df['Embarked'])
df['Cabin_Code'] = le.fit_transform(df['Cabin'])

# (Feature Engineering)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# (Outliers)
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df['Age'] >= lower) & (df['Age'] <= upper)]

print(f"Final Row Count after Cleaning: {len(df)}")


sns.set_theme(style="whitegrid")


plt.figure(figsize=(8, 5))
df['Sex_Label'] = df['Sex'].str.capitalize()
sns.countplot(x='Sex_Label', hue='Survived', data=df, palette='magma')
plt.title("Survival Count by Gender (Original Data)")
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=20, kde=True, color='teal')
plt.title("Age Distribution (After Preprocessing)")
plt.show()

# relation between ticket and survived
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Survived', data=df, palette='viridis')
plt.title("Survival Rate by Class")
plt.show()

print("Survival Percentages:")
print(df.groupby('Sex_Label')['Survived'].mean() * 100)



df.to_csv(r'D:\\Titanic_Cleaned_Final.csv', index=False)
print("Done! Your clean file is saved at D:\\Titanic_Cleaned_Final.csv")
