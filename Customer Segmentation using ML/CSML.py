import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Loading data
df = pd.read_csv("Tab_Customers_Mall.csv")
df.head
df.info
df.tail

#Gender distribution between males and females of Mall
genders = df.Gender.value_counts()
sns.barplot(x=genders.index, y=genders.values)
plt.show()

#Visualization the customers with different age
age18_25 = df.Age[(df.Age<=25)&(df.Age>=18)]
age26_35 = df.Age[(df.Age<=35)&(df.Age>=26)]
age36_45 = df.Age[(df.Age<=45)&(df.Age>=36)]
age46_55 = df.Age[(df.Age<=55)&(df.Age>=46)]
age55above = df.Age[(df.Age>=56)]

x=["18-25", "26-35", "36-45", "46-55", "Above55"]
y=[len(age18_25.values), len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age55above.values)]

plt.figure(figsize=(10, 5)) #size object view
plt.title=("Number of customers and ages")
plt.xlabel=("Ages")
plt.ylabel=("Number of customers")
sns.barplot(x=x, y=y)
plt.show()

#Visualization the highest spending scores among the customers
ss1_20= df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=1) &(df["Spending Score (1-100)"]<=20)]
ss21_40= df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=21) &(df["Spending Score (1-100)"]<=40)]
ss41_60= df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=41) &(df["Spending Score (1-100)"]<=60)]
ss61_80= df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=61) &(df["Spending Score (1-100)"]<=80)]
ss81_100= df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=81) &(df["Spending Score (1-100)"]<=100)]

x=["1-20","21-40","41-60","61-80","81-100"]
y=[len(ss1_20.values), len(ss21_40.values),len(ss41_60.values),len(ss61_80.values),len(ss81_100.values)]

plt.figure(figsize=(8,4)) #size object view
sns.barplot(x=x , y=y)
plt.title=("Spending scores of the customers")
plt.xlabel=("Spending Scores")
plt.ylabel=("score of customers")
plt.show()

#Visualization the annual income of the customers.
#Present the Annual Income of the customers on the X-axis and the number of customers on the Y-axis
ai0_30 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=0)&(df["Annual Income (k$)"]<=30)]
ai31_60 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=31)&(df["Annual Income (k$)"]<=60)]
ai61_90 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=61)&(df["Annual Income (k$)"]<=90)]
ai91_120 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=91)&(df["Annual Income (k$)"]<=120)]
ai121_150 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=121)&(df["Annual Income (k$)"]<=150)]

x=["0-30","31-60","61-90","91-120","121-150"]
y=[len(ai0_30.values), len(ai31_60.values), len(ai61_90.values),len(ai91_120.values), len(ai121_150.values)]

plt.figure(figsize=(10,6))
sns.barplot(x=x,y=y,)
#plt.title=("Annual Income of customers")
#plt.xlabel("Annual Income (k$)")
#plt.ylabel("Number of customers")
plt.show()