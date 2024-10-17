# -*- coding: utf-8 -*-
"""
Created on Ekim 2024
@author: Assoc Prof. Elif Kartal & Prof. Dr. M. Erdal Balaban
@title: Data Visualization
@dataset: Health Insurance https://www.kaggle.com/bmarco/health-insurance-data 

"""

# pip install numpy
# pip install pandas
# pip install matplotlib
# pip install seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


insurance = pd.read_csv("insurance.csv")


insurance.dtypes
insurance["sex"] = insurance["sex"].astype("category")
insurance["smoker"] = insurance["smoker"].astype("category")
insurance["region"] = insurance["region"].astype("category")
insurance.dtypes



insurance.describe() 

pd.set_option("display.max_columns", 20)
insurance.describe(include="all") 

## LINE PLOT
# Example - 1: Age of the first 10 customers
x = range(1,11)
y = insurance.iloc[0:10,0]
plt.plot(x, y, "o:r")

plt.plot(x, y, "^k:")

plt.plot(x, y, 
         linestyle = "dashed", 
         color="hotpink",
         linewidth = "5")


# For the other parameters please see
help(plt.plot)

# Example - 2: BMI comparison of customers in the first and second 20
x1 = np.arange(1,21)
y1 = insurance.iloc[0:20,2]
y2 = insurance.iloc[20:40,2]
plt.xticks(x1)
plt.plot(x1,y1,x1,y2)

# Example - 3
x = np.arange(1,51)
y = insurance.iloc[0:50,0]
plt.title("Age of The First 50 Customers") 
plt.xlabel("ID") 
plt.ylabel("Age") 
plt.plot(x, y)
plt.grid(color = "red", linestyle = '--', linewidth = 0.5)
# plt.grid(axis = "x") 
# plt.grid(axis = "y") 


# SUB GRAPHS
# plot 1: BMI of the first 20 customers
x = np.arange(1,21)
y1 = insurance.iloc[0:20,2]

plt.subplot(1, 2, 1)
#plt.subplot(2, 1, 1)
plt.plot(x,y1)
plt.xticks(x)
plt.title("The First 20 Customers")

#plot 2: BMI of the second 20 customers
y2 = insurance.iloc[20:40,2]

plt.subplot(1, 2, 2)
#plt.subplot(2, 1, 2)
plt.plot(x,y2)
plt.xticks(x)

plt.title("The first 20 customers")

plt.suptitle("BMI Comparison")
plt.tight_layout(pad=1)


## BAR PLOT
mySummary = insurance.groupby("smoker")["charges"].mean()

def addLbls(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = "center")

# Example - 1

mySummary = insurance.groupby("smoker")["charges"].mean()

plt.bar(x=mySummary.index, height=mySummary.values, color="b")
plt.xlabel("Smoking")
plt.ylabel("Charges")
addLbls(mySummary.index, mySummary.values.round(2))

# Example - 2:
plt.barh(y=mySummary.index, width=mySummary.values, color="g")

# Example - 3
plt.bar(x=mySummary.index, height=mySummary.values, color="b", width = 0.95)

# Example - 4
plt.barh(y=mySummary.index, width=mySummary.values, color="r", height=0.2)

## PIE CHART

# Example - 1
mySummary = insurance.groupby("region")["charges"].sum()

labels = mySummary.index
vals = mySummary.values
myCols = sns.color_palette("viridis", 4)
mySelection = (0, 0, 0.2, 0)

myCols = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'] 

plt.pie(vals, 
        explode=mySelection, 
        labels=labels, 
        autopct="%4.1f%%",  
        shadow=True, 
        startangle=180,
        colors=myCols)

# plt.legend()
# plt.legend(title="Regions")


## SCATTER PLOT

# Example - 1:
x = insurance.age
y = insurance.bmi
plt.scatter(x, y)
plt.xlabel("Age")
plt.ylabel("BMI")


mySelection = [1072, 548, 1032, 437, 154, 653, 645, 862, 25, 603]
mySubset = insurance.iloc[mySelection,[0,6]]
x = mySubset.age
y = mySubset.charges
plt.scatter(x, y)
plt.xlabel("Age")
plt.ylabel("Charges")




# Example - 2: 
sns.scatterplot(x="age", 
                y="charges",
                hue="smoker",
                data=insurance)

# Example - 3
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection ="3d")
ax.scatter(insurance.age, insurance.charges, insurance.bmi, color="darkblue")
ax.set_xlabel("Age")
ax.set_ylabel("Charges")
ax.set_zlabel("BMI", rotation=90)
ax.zaxis.labelpad=-0.7
plt.title("3D-Scatter Plot")


# # Example - 4
pd.plotting.scatter_matrix(insurance,figsize=(20,20),grid=True, marker='o')


# For more info about colormaps, please see: https://matplotlib.org/stable/tutorials/colors/colormaps.html

## HISTOGRAM

# Example - 1
sns.histplot(data=insurance, x="bmi", color="magenta")

# Example - 2
sns.histplot(data=insurance, y="bmi", color="lime")

# Example - 3
sns.histplot(data=insurance, x="bmi", color="plum", bins=8)

# Example - 4
sns.histplot(data=insurance, x="bmi", color="salmon", bins=12)

# Example - 5
sns.histplot(data=insurance, x="bmi", bins=8, color="teal", kde=True)

# Example - 6
sns.histplot(x="bmi", kde = True, data = insurance, color="purple")


# BOX PLOT

# Example - 1

insurance["bmi"].describe().round(2)
bp = plt.boxplot(insurance["bmi"], 
                 vert=True, 
                 showmeans=True,
                 meanline=True,
                 labels=('x'),
                 patch_artist=True,
                 medianprops={'linewidth': 2, 'color': 'blue'},
                 meanprops={'linewidth': 2, 'color': 'magenta'},
                 notch=True)
plt.ylabel("Charges")
plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'])


# Example - 2
sns.boxplot(y="smoker", 
            x="charges", 
            data=insurance, 
            palette="rainbow")


# VIOLIN GRAPH

# Example - 1
sns.violinplot(y="smoker", 
               x="charges", 
               data=insurance, 
               palette="coolwarm")


# HEAT MAP

# Example - 1
myCors = insurance[["age", "bmi", "children", "charges"]].corr()
sns.heatmap(
    myCors, 
    annot = True,
    square=True,
    cmap="Reds"
)