# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:10:01 2024

@author: test2
"""

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
  
# metadata 
print(iris.metadata) 
  
# variable information 
print(iris.variables) 
import pandas as pd
iris = pd.read_csv("iris.data");
