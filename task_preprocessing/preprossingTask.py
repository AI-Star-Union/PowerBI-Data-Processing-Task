import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

df = pd.read_csv(r"D:\\Test.csv")

print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())

#handle missing values
df['Age']=df['Age'].fillna(df['Age'].mean())

df['Age']=df['Age'].astype(int)

df['Cabin']=df['Cabin'].fillna("unknown")

df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare']=df['Fare'].fillna(df['Fare'].mean())

# delete dublicates data
df.drop_duplicates( inplace=True)

le_sex = LabelEncoder()
df['Sex'] = df['Sex'].map({'male':1,'female':0})

le_embarked = LabelEncoder()
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

le_cabin = LabelEncoder()
df['Cabin'] = le_cabin.fit_transform(df['Cabin'])

# add new column
df['FamilySize']=df['SibSp']+df['Parch']
print(df.head())

# check the outliers
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(lower, upper)

print("missing value after preprocessing :")
print(df.isnull().sum())


#visualization
print(df['Sex'].unique())
# number of girl and boys

sns.countplot(x='Sex', data=df)
plt.title("Male vs Female")
plt.show()


# number of survival
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()



# relationship between sex and survival
sns.countplot(x='Sex', hue='Survived', data=df)
plt.show()

# Age Distribution
plt.hist(df['Age'], bins=20)
plt.title("Age Distribution")
plt.show()



df.to_csv(r'D:\\TestAfterProcessing.csv')
