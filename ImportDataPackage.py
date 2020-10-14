"""

@author: Gene Kindberg-Hanlon, genekindberg @ googlemail .com
Draft code - please report errors, questions and suggestions. 
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import DFM as DF

###########################################################################################
# Very simple data example of nowcasting US quarterly GDP using 3 factors (of only two series each) using FRED data.
# Most nowcasting programs will use PMIs and other proprierary series that cannot be posted publicaly
# This code will also work with 

#os.chdir('/home/autumn/Documents/Nowcasting/MIDAS')
dirname = os.path.dirname(__file__)
#concatenates your current directory with your desired subdirectory
excel_file = os.path.join(dirname, r'ImportFile.xlsx')

####################################################
# Import data
GDP_dat = pd.read_excel(excel_file, sheet_name='GDP')
Emp_dat = pd.read_excel(excel_file, sheet_name='Emp')
Cons_dat = pd.read_excel(excel_file, sheet_name='Cons')
IPExp_dat = pd.read_excel(excel_file, sheet_name='IPExp')



GDP = GDP_dat['GDP'][0:].values.reshape(-1,1) # GDP data for nowcasting
Dates = GDP_dat.Date

                        ## Make list of pandas monthly frames
MonthlyDat = [Emp_dat.drop(columns=['Date']), 
                        Cons_dat.drop(columns='Date'), IPExp_dat.drop(columns='Date')]

MAterm = 0 # turn on or off the errors in the observation equations (Var-Cov R) taking an MA specification. (eps_r = rho*eps_e(t-1)+eps)
lags = 6 # state transition equation number of lags
lagsH = 1 # Number of lags to use in observation equation (of the factors)
K = 3 # number of non GDP factors, (entries in monthlydat)
Qs = 1 # number of GDP factors - Do not adjust number of GDP series - possibly this will be adapted in the future - only works with one for now.
normGDP = 1 # estimate model with a normalized GDP series
# Initialise the factor model
DynamicFac = DF.DynamicFactorModel(GDP, MonthlyDat, Dates, K, Qs, lags, lagsH, MAterm, normGDP)



burn = 50 # Number of throw-away initial draws (In practice may burn 100 and save 1000, but will take longer)
save = 50 # Gibbs draws to save
DynamicFac.estimateGibbs(burn, save) # Estimate model using data to last quarterly GDP datapoint
DynamicFac.Nowcast(2008, 2) # Year to start nowcasting, how many quarters to nowcast

print('Plot quasi-out of sample month 2 nowcasts against data (these use full estimation sample to estimate parameters and previous quarter state)')
DynamicFac.PlotFcast(2) # Plot forecast 

np.set_printoptions(precision=3)

print('View RMSEs, as current quarter, next quarter (columns), and months (1,2,3) of the quarter (rows).')
print(DynamicFac.RMSE)  
print('Series of current quarter nowcasts (quarters rows, columns month of each quarter)')
print(DynamicFac.Fcast_current) 
print('Series of next quarter nowcasts (quarters rows, columns month of each quarter)')
print(DynamicFac.Fcast_next)

print('GDP outturn')
print(DynamicFac.Outturn_current)


print('Dates series')
print(DynamicFac.Datesaug) 