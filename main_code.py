
# coding: utf-8

# In[2]:

#get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm


# In[51]:
#Delete the first row of dataset file
dataset = pd.read_csv('data/HT_Sensor_dataset.dat',sep = '  ',header = None,engine='python')
dataset.columns = ['id','time','R1','R2','R3','R4','R5','R6','R7','R8','Temp.','Humidity']
dataset.set_index('id',inplace = True)
print(dataset.head())


# In[55]:
#Delete the first row of metadata file
output = pd.read_csv('data/HT_Sensor_metadata.dat',sep = '\t',header = None)
output.columns = ['id','date','class','t0','dt']
print(output.head())


# # Joining Dataset
# 
# We can see that the two dataset are for same experiment. So, we need to join the two datasets on id. 

# In[57]:

dataset = dataset.join(output,how = 'inner')
dataset.set_index(np.arange(dataset.shape[0]),inplace = True)
dataset['time']  += dataset['t0']
dataset.drop(['t0'],axis = 1,inplace=True)
dataset.head()


# Now, we are going to see the plots of reading of the sensors with time.  
# The graph shown represent day 17 reading.  
# As we can see from the graphs  sensor R7 shows minimum reading and sensor R1 has maximum reading.  
# The readings for rest of the days will be similar to the readings of plots shown below if the temprature and humidity are similar.

# In[59]:

fig, axes = plt.subplots(nrows=3, ncols=2)#, sharex=True, sharey=True)
fig.set_figheight(20)
fig.set_figwidth(25)
fig.subplots_adjust(hspace=.5)

axes[0,0].plot(dataset.time[dataset.id == 3],dataset.R1[dataset.id == 3],c = 'red',linewidth = '2.0')
axes[0,0].set_title('R1 Vs Time')
axes[0,0].set_xlabel('Time(hour)')
axes[0,0].set_ylabel('R1_values(kilo ohm)')

axes[0,1].plot(dataset.time[dataset.id == 3],dataset.R2[dataset.id == 3],c = 'green',linewidth = '2.0')
axes[0,1].set_title('R2 Vs Time')
axes[0,0].set_xlabel('Time(hour)')
axes[0,0].set_ylabel('R2_values (kilo ohm)')


axes[1,0].plot(dataset.time[dataset.id == 3],dataset.R3[dataset.id == 3],c = 'orange',linewidth = '2.0',label = 'R3 (Sensor)')
#axes[1,0].set_title('R3 Vs Time')
axes[1,0].set_xlabel('Time(hour)')
axes[1,0].set_ylabel('R3_values (kilo ohm)')


axes[1,0].plot(dataset.time[dataset.id == 3],dataset.R4[dataset.id == 3],c = 'blue',linewidth = '2.0',label = 'R4')
axes[1,0].set_title('R4 and R3 Vs Time')
axes[1,0].set_xlabel('Time(hour)')
axes[1,0].set_ylabel('Reading (kilo ohm)')
axes[1,0].legend(loc = 4)

axes[1,1].plot(dataset.time[dataset.id == 3],dataset.R5[dataset.id == 3],c = 'pink',linewidth = '2.0')
axes[1,1].set_title('R5 Vs Time')
axes[1,1].set_xlabel('Time(hour)')
axes[1,1].set_ylabel('R5_values (kilo ohm)')
 

axes[2,0].plot(dataset.time[dataset.id == 3],dataset.R6[dataset.id == 3],c = 'violet',linewidth = '2.0',label = 'R6')
#axes[2,0].set_title('R6 Vs Time')
axes[2,0].set_xlabel('Time(hour)')
axes[2,0].set_ylabel('R6_values (kilo ohm)')


axes[2,0].plot(dataset.time[dataset.id == 3],dataset.R7[dataset.id == 3],c = 'black',linewidth = '2.0',label ='R7')
axes[2,0].set_title('R7 and R6 Vs Time')
axes[2,0].set_xlabel('Time(hour)')
axes[2,0].set_ylabel('Reading (kilo ohm)')
axes[2,0].legend()

axes[2,1].plot(dataset.time[dataset.id == 3],dataset.R8[dataset.id == 3],c = 'brown',linewidth = '2.0')
axes[2,1].set_title('R8 Vs Time')
axes[2,1].set_xlabel('Time(hour)')
axes[2,1].set_ylabel('R8_values (kilo ohm)')
plt.suptitle('Sensor Reading on Day 3')
pl.savefig("Graph1.png", dpi=300)


# Now, the above reading will be similar for all days if the Humidity and Temprature are similar.  
# Let us plot the Humidity and Temprature vs Time.

# In[60]:

fig, axes = plt.subplots(nrows=1, ncols=2)#, sharex=True, sharey=True)
fig.set_figheight(5)
fig.set_figwidth(20)
fig.subplots_adjust(hspace=.5)

axes[0].plot(dataset.time[dataset.id == 17],dataset['Temp.'][dataset.id == 17],c = 'r')
axes[0].set_title('R1 Vs Temp')
axes[0].set_xlabel('Time (hour)')
axes[0].set_ylabel('Temprature (C)')
axes[1].plot(dataset.time[dataset.id == 17],dataset.Humidity[dataset.id == 17],c = 'green')
axes[1].set_title('R2 Vs Humidity')
axes[1].set_xlabel('Humidity (%)')
plt.suptitle('Temprature Reading on Day 3')
pl.savefig("Graph2.png", dpi=300)


# In[7]:

dataset.corr()


# In[8]:

dataset.corr() > 0.98

