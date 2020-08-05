import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import statsmodels.api as sm

metadata_filename = '../data/Final_HT_Sensor_metadata.dat'
dataset_filename = '../data/Final_HT_Sensor_dataset.dat'

banana_id = 17
wine_id = 19

metadata = np.loadtxt( metadata_filename, dtype=str)
metadata_aux = np.array( metadata[:,[0,3,4]], dtype=float )

banana_info = metadata_aux[banana_id]
bt0 = banana_info[1]
btf = bt0 + banana_info[2]

wine_info = metadata_aux[wine_id]
wt0 = wine_info[1]
wtf = wt0 + wine_info[2]

dataset = np.loadtxt( dataset_filename )

bData = dataset[dataset[:,0] == banana_id,1:]
bData[:,0] += bt0

wData = dataset[dataset[:,0] == wine_id,1:]
wData[:,0] += wt0


pl.figure( figsize = (9,7) )
gs1 = gridspec.GridSpec( 6,2 )
gs1.update( wspace=0.4, hspace=0.0 )

ax = {}
for j in range(6):
    ax[j,0] = pl.subplot( gs1[j*2] )
    ax[j,1] = pl.subplot( gs1[j*2+1] )


ax[0,0].plot( bData[:,0], bData[:,10], color=(0.1,0.8,0.1), lw=1.5 )
ax[0,0].set_ylim(55,65.4)
ax[0,1].plot( wData[:,0], wData[:,10], color=(0.1,0.8,0.1), lw=1.5 )
ax[0,1].set_ylim(55,65.4)
ax[0,0].set_yticks( np.arange(56.,65.5,3) )
ax[0,1].set_yticks( np.arange(56.,65.5,3) )

ax[1,0].plot( bData[:,0], bData[:, 9], color=(1.0,0.1,0.0), lw=1.5 )
ax[1,0].set_ylim(26.1,29.9)
ax[1,1].plot( wData[:,0], wData[:, 9], color=(1.0,0.1,0.0), lw=1.5 )
ax[1,1].set_ylim(26.1,29.9)
ax[1,0].set_yticks( np.arange(27.,29.5,1) )
ax[1,1].set_yticks( np.arange(27.,29.5,1) )

ax[2,0].plot( bData[:,0], bData[:, 1], color=(0.3,0.3,0.3), lw=1.5)
ax[2,0].plot( bData[:,0], bData[:, 4], '-', color=(1.0,0.5,0.1), lw=1.5 )
ax[2,0].set_ylim(7.2,13.9)
ax[2,1].plot( wData[:,0], wData[:, 1], color=(0.3,0.3,0.3), lw=1.5, label=r"$R_1$"  )
ax[2,1].plot( wData[:,0], wData[:, 4], color=(1.0,0.5,0.1), lw=1.5, label=r"$R_4$"  )
ax[2,1].set_ylim(7.2,13.9)
ax[2,0].set_yticks( np.arange(8.,12.5,2) )
ax[2,1].set_yticks( np.arange(8.,12.5,2) )

ax[2,1].legend(frameon = False, fontsize=12, bbox_to_anchor=(-0.08,0.9), handletextpad=0)

ax[3,0].plot( bData[:,0], bData[:, 2], '--', color=(0.3,0.3,0.3), lw=1.5, zorder=3 )
ax[3,0].plot( bData[:,0], bData[:, 3], color=(1.0,0.5,0.1), lw=1.5, zorder=1 )
ax[3,0].set_ylim(3.2,12.9)
ax[3,1].plot( wData[:,0], wData[:, 2], color=(0.3,0.3,0.3), lw=1.5, label=r"$R_2$" )
ax[3,1].plot( wData[:,0], wData[:, 3], color=(1.0,0.5,0.1), lw=1.5, label=r"$R_3$"  )
ax[3,1].set_ylim(3.2,12.9)
ax[3,0].set_yticks( np.arange(5.,12.5,2 ) )
ax[3,1].set_yticks( np.arange(5.,12.5,2) )

ax[3,1].legend(frameon = False, fontsize=12, bbox_to_anchor=(-0.08,0.9), handletextpad=0)

ax[4,0].plot( bData[:,0], bData[:, 5], '-', color=(0.3,0.3,0.3), lw=1.5 )
ax[4,0].plot( bData[:,0], bData[:, 6], '-', color=(1.0,0.5,0.1), lw=1.5 )
ax[4,0].set_ylim(3.0,15)
ax[4,1].plot( wData[:,0], wData[:, 5], '-', color=(0.3,0.3,0.3), lw=1.5, label=r"$R_5$" )
ax[4,1].plot( wData[:,0], wData[:, 6], '-', color=(1.0,0.5,0.1), lw=1.5, label=r"$R_6$" )
ax[4,1].set_ylim(3.,15)
ax[4,0].set_yticks( np.arange(4.,12.5,4) )
ax[4,1].set_yticks( np.arange(4.,12.5,4) )

ax[4,1].legend(frameon = False, fontsize=12, bbox_to_anchor=(-0.08,0.9), handletextpad=0)

ax[5,0].plot( bData[:,0], bData[:, 7], '-', color=(0.3,0.3,0.3), lw=1.5 )
ax[5,0].plot( bData[:,0], bData[:, 8], '-', color=(1.0,0.5,0.1), lw=1.5 )
ax[5,0].set_ylim(1.1,6.8)
ax[5,1].plot( wData[:,0], wData[:, 7], '-', color=(0.3,0.3,0.3), lw=1.5, label=r"$R_7$" )
ax[5,1].plot( wData[:,0], wData[:, 8], '-', color=(1.0,0.5,0.1), lw=1.5, label=r"$R_8$" )
ax[5,1].set_ylim(1.1,6.8)
ax[5,0].set_yticks( np.arange(2.,6.5,2) )
ax[5,1].set_yticks( np.arange(2.,6.5,2) )

ax[5,1].legend(frameon = False, fontsize=12, bbox_to_anchor=(-0.08,0.9), handletextpad=0)


for j in range(6):
    ax[j,0].plot( [bt0,bt0,btf,btf], [-100,100,100,-100], '-',
                  color=(0.1,0.1,1.0), lw=2.0, alpha = 0.5 )
    ax[j,1].plot( [wt0,wt0,wtf,wtf], [-100,100,100,-100], '-',
                  color=(0.1,0.1,1.0), lw=2.0, alpha = 0.5 )
    ax[j,0].set_xticks(np.arange(6.0,8,0.25))
    ax[j,0].set_xlim(6.0, bData[:,0].max())
    ax[j,0].set_xticklabels([])
    ax[j,1].set_xticks(np.arange(22,23.8,0.25))
    ax[j,1].set_xlim(wData[:,0].min(), wData[:,0].max())
    ax[j,1].set_xticklabels([])


ax[0,0].set_title("Banana")
ax[0,1].set_title("Wine")

ax[0,0].set_ylabel(r"$H$ (%)")
ax[1,0].set_ylabel(r"$T_E$ (C)")
ax[2,0].set_ylabel(r"$R_{1,4}$ (k$\Omega$)")
ax[3,0].set_ylabel(r"$R_{2,3}$ (k$\Omega$)")
ax[4,0].set_ylabel(r"$R_{5,6}$ (k$\Omega$)")
ax[5,0].set_ylabel(r"$R_{7,8}$ (k$\Omega$)")

ax[5,0].set_xticklabels(['6.0','','6.5','','7.0','','7.5'])
ax[5,1].set_xticklabels(['22.0','','22.5','','23.0','','23.5'])
ax[5,0].set_xlabel("Time (h)")
ax[5,1].set_xlabel("Time (h)")


pl.savefig( "Final_Plot_Graph.png", dpi=300 )

pl.clf()

#Loading the data again in different library for Training & Testing the prediction
dataset_data = pd.read_csv(dataset_filename ,sep = '  ',header = None,engine='python')
dataset_data.columns = ['id','time','R1','R2','R3','R4','R5','R6','R7','R8','Temp.','Humidity']
dataset_data.set_index('id',inplace = True)
dataset_data.head()

dataset_meta = pd.read_csv(metadata_filename,sep = '\t',header = None)
dataset_meta.columns = ['id','date','class','t0','dt']
dataset_meta.head()


# Joining metadata with dataset table
dataset_train = dataset_data.join(dataset_meta,how = 'inner')
dataset_train.set_index(np.arange(dataset_train.shape[0]),inplace = True)
dataset_train['time']  += dataset_train['t0']
dataset_train.drop(['t0'],axis = 1,inplace=True)
dataset_train.head()

dataset_train.corr()
dataset_train.corr() > 0.98


#Splitting the data into Train & Test data so that we can train and test the prediction
xtrain,xtest,ytrain,ytest = train_test_split(dataset_train[[u'R1',u'R2',u'R3',u'R4',u'R5',u'R6',u'R7',u'R8',u'Temp.',u'Humidity']].values,dataset_train['class'].values,train_size = 0.25)
#Above line contains the train data set percentage
for i in range(ytrain.shape[0]):
    if(ytrain[i] == 'background'):
        ytrain[i] = 0
    elif(ytrain[i] == 'banana'):
        ytrain[i] = 1
    else:
        ytrain[i] = 2

for i in range(ytest.shape[0]):
    if(ytest[i] == 'background'):
        ytest[i] = 0
    elif(ytest[i] == 'banana'):
        ytest[i] = 1
    else:
        ytest[i] = 2

ytrain = ytrain.astype('int64')
ytest = ytest.astype('int64')


#Applying RLM to get the summary
xtrain_dataframe = pd.DataFrame(xtrain)
ytrain_dataframe = pd.DataFrame(ytrain)
xtest_dataframe = pd.DataFrame(xtest)
ytest_dataframe = pd.DataFrame(ytest)
xtrain_dataframe.columns = [u'R1',u'R2',u'R3',u'R4',u'R5',u'R6',u'R7',u'R8',u'Temp.',u'Humidity']
ytrain_dataframe.columns = ['class']
xtest_dataframe.columns = [u'R1',u'R2',u'R3',u'R4',u'R5',u'R6',u'R7',u'R8',u'Temp.',u'Humidity']
ytest_dataframe.columns = ['class']
res = sm.RLM(ytrain_dataframe, xtrain_dataframe).fit()
print(res.summary())


#Scaling the Data so that Logistic Regression & SVM can be applied
scaler = StandardScaler()
xtrain_scaled = scaler.fit_transform(xtrain)


#Logistic Regression Logic
print('Logistic Regression related Results')
Cs = [0.001,0.01,0.1,1,10,100]
score_test = []
score_train = []
kf = KFold(n_splits=11)
itr_kf = kf.get_n_splits(xtrain_scaled.shape[0])
for c in Cs:
    score1 = []
    score2 = []
    est = LogisticRegression(C=c,n_jobs= 8)
    for itrain,itest in kf.split(xtrain_scaled):
        est.fit(xtrain_scaled[itrain],ytrain[itrain])
        score1.append(accuracy_score(est.predict(xtrain_scaled[itest]),ytrain[itest]))
        score2.append(accuracy_score(est.predict(xtrain_scaled[itrain]),ytrain[itrain]))
    score_test.append(np.mean(score1))
    score_train.append(np.mean(score2))

plt.plot([0.001,0.01,0.1,1,10,100],score_train,'o-',label = 'training_score')
plt.plot([0.001,0.01,0.1,1,10,100],score_test,'o-',label = 'testing_score')
plt.xscale('log')
plt.legend(loc = 4)
plt.xlabel('Values of Regularization Parameter')
plt.ylabel('Accuracy Score')
plt.axhline(y=score_train[1],c = 'black')
plt.title('Prediction Acurracy of Logistic Regression')
pl.savefig("PredictionAccuracyOfLogisticRegression.png", dpi=300)
pl.clf();#clearing the graph

est = LogisticRegression(C = Cs[np.argmax(score_train)],n_jobs = 8)
est.fit(xtrain_scaled,ytrain)
ypred = est.predict(xtest)

print('accuracy_score of LogisticRegression')
print(accuracy_score(ypred, ytest))

print('confusion matrix is:')
print(confusion_matrix(ytest, ypred))


#SVM Logic to Predict from the Trained Data
print('SVM Logic Related Results')
C_2d_range = [1e-2]
gamma_2d_range = [1e-1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        print('SVC done')
        clf.fit(xtrain, ytrain)
        print('fit done')
        classifiers.append((C, gamma, clf))
print('confusion matrix is:')
svm_ypred = clf.predict(xtest)
print(confusion_matrix(ytest, svm_ypred))
print(accuracy_score(svm_ypred, ytest))
