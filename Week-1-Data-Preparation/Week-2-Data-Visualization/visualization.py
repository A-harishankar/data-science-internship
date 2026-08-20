import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

titanic = pd.read_csv("../dataset/train.csv")

sns.countplot(x="Survived", data=titanic)

plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Number of Passengers")

plt.show()

sns.countplot(x="Sex", hue="Survived", data=titanic)

plt.title("Survival Count by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.show()

gender_survival = titanic.groupby("Sex")["Survived"].mean() * 100

plt.figure(figsize=(8, 5))

sns.barplot(
    x=gender_survival.index,
    y=gender_survival.values
)

plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate (%)")
plt.ylim(0, 100)

for i, value in enumerate(gender_survival):
    plt.text(
        i,
        value + 2,
        f"{value:.1f}%",
        ha="center"
    )

plt.show()
sns.countplot(
    x="Pclass",
    hue="Survived",
    data=titanic
)

plt.title("Survival Count by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")

plt.show()

class_survival = titanic.groupby("Pclass")["Survived"].mean() * 100

plt.figure(figsize=(8, 5))

sns.barplot(
    x=class_survival.index,
    y=class_survival.values
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.ylim(0, 100)

for i, value in enumerate(class_survival):
    plt.text(
        i,
        value + 2,
        f"{value:.1f}%",
        ha="center"
    )

plt.show()

