import pandas as pd
import numpy as np
import math

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt # default library for making plots
from matplotlib import colors as mcolors 

import seaborn as sns ; sns.set()# for making prettier plots!

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.mixture import GaussianMixture

plt.rcParams['figure.figsize'] = [24, 12]

col_list = [
 'Calories Burned',
 'Calories BMR',
 'Steps',
 'Distance (Km)',
 'Resting Heart Rate',
 'Minutes Sedentary',
 'Minutes Lightly Active',
 'Minutes Fairly Active',
 'Minutes Very Active',
 'Activity Calories',
 'Fat Burn minutes',
 'Minutes Asleep',
 'Minutes REM sleep']

base_name = 'patient'

num_patients_1 = 40
num_patients_2 = 60
num_patients_3 = 10
num_patients_4 = 130
num_patients_5 = 100
num_patients_6 = 100

patient_list = []
patient_list_1 = []
patient_list_2 = []
patient_list_3 = []
patient_list_4 = []
patient_list_5 = []
patient_list_6 = []

for j in range(0, num_patients_1):
    index_name_1 = base_name + str(11000) + str(j)
    patient_list_1.append(index_name_1)

for k in range(0, num_patients_2):
    index_name_2 = base_name + str(11000) + str(num_patients_1) + str(k)
    patient_list_2.append(index_name_2)

for l in range(0, num_patients_3):
    index_name_3 = base_name + str(11000) + str(num_patients_2) + str(l)
    patient_list_3.append(index_name_3)

for m in range(0, num_patients_4):
    index_name_4 = base_name + str(11000) + str(num_patients_3) + str(m)
    patient_list_4.append(index_name_4)

for n in range(0, num_patients_5):
    index_name_5 = base_name + str(11000) + str(num_patients_4) + str(n)
    patient_list_5.append(index_name_5)

for o in range(0, num_patients_6):
    index_name_6 = base_name + str(11000) + str(num_patients_5) + str(o)
    patient_list_6.append(index_name_6)

patient_list = patient_list_1 + patient_list_2 + patient_list_3 + patient_list_4 + patient_list_5 + patient_list_6
big_boy = pd.DataFrame(columns=col_list, index = patient_list)

#Average calorie burn rates. Normal, and other distributions will be applied later
r_light = 1.95
r_med = 4.75
r_heavy = 8.8
r_sed = 0.03
r_dist = 1.95

#Averages
walk_ave = 3.5

for each in patient_list_1:

    #Average calorie burn rates. Normal, and other distributions will be applied later
    r_light = 1.95
    r_med = 4.75
    r_heavy = 8.8
    r_sed = 0.03
    r_dist = 1.95

    #Averages
    walk_ave = 3.5

    #Data corresponding to steps and distance walked
    distance = abs(np.random.uniform(0.2,5))
    steps = distance*(np.random.randint(1200,1800))

    #Data corresponding to time spent on activity
    t_light = abs(np.random.normal(250, 100))
    t_med = (abs(np.random.uniform(0.1,0.5)))*t_light
    t_heavy = (abs(np.random.uniform(0,0.9)))*t_med
    t_sed = abs(np.random.normal(1000, 300))
    t_far_burn = t_heavy*(abs(np.random.uniform(0.75,1.25)))

    #Data corresponding to sleep
    t_asleep = abs(np.random.uniform(5.5,8.5))*60
    t_rem = (abs(np.random.uniform(0.15,0.3)))*t_asleep

    #Resting Heart Rate
    rhr = (np.random.randint(60,100))

    Activity_cals = (t_light)*(r_light)+(t_med)*(r_med)+(t_heavy)*(r_heavy) + (t_sed)*(r_sed)

    Cals_burned = Activity_cals + distance*(r_dist) + np.random.randint(100,200)

    Cals_BMR = (abs(np.random.uniform(0.5,0.85)))*(Cals_burned)

    big_boy.loc[each] = pd.Series({ 'Calories Burned':Cals_burned,
 'Calories BMR':Cals_BMR,
 'Steps':steps,
 'Distance (Km)': distance,
 'Resting Heart Rate':rhr,
 'Minutes Sedentary':t_sed,
 'Minutes Lightly Active':t_light,
 'Minutes Fairly Active':t_med,
 'Minutes Very Active':t_heavy,
 'Activity Calories':Activity_cals,
 'Fat Burn minutes':t_far_burn,
 'Minutes Asleep':t_asleep,
 'Minutes REM sleep':t_rem})

for each in patient_list_2:

    #Average calorie burn rates. Normal, and other distributions will be applied later
    r_light = 2.5
    r_med = 5.1
    r_heavy = 9.8
    r_sed = 0.03
    r_dist = 2.15

    #Averages
    walk_ave = 2

    #Data corresponding to steps and distance walked
    distance = abs(np.random.uniform(walk_ave,9))
    steps = distance*(np.random.randint(1500,1900))

    #Data corresponding to time spent on activity
    t_light = abs(np.random.normal(450, 200))
    t_med = (abs(np.random.uniform(0.2,0.6)))*t_light
    t_heavy = (abs(np.random.uniform(0.3,1.1)))*t_med
    t_sed = abs(np.random.normal(2000, 500))
    t_far_burn = t_heavy*(abs(np.random.uniform(0.9,1.2)))

    #Data corresponding to sleep
    t_asleep = abs(np.random.uniform(7.5,9.5))*60
    t_rem = (abs(np.random.uniform(0.2,0.3)))*t_asleep

    #Resting Heart Rate
    rhr = (np.random.randint(55,70))

    Activity_cals = (t_light)*(r_light)+(t_med)*(r_med)+(t_heavy)*(r_heavy) + (t_sed)*(r_sed)
    Cals_burned = Activity_cals + distance*(r_dist) + np.random.randint(100,200)
    Cals_BMR = (abs(np.random.uniform(0.6,0.9)))*(Cals_burned)

    big_boy.loc[each] = pd.Series({ 'Calories Burned':Cals_burned,
'Calories BMR':Cals_BMR,
'Steps':steps,
'Distance (Km)': distance,
'Resting Heart Rate':rhr,
'Minutes Sedentary':t_sed,
'Minutes Lightly Active':t_light,
'Minutes Fairly Active':t_med,
'Minutes Very Active':t_heavy,
'Activity Calories':Activity_cals,
'Fat Burn minutes':t_far_burn,
'Minutes Asleep':t_asleep,
'Minutes REM sleep':t_rem})


for each in patient_list_3:

    #Average calorie burn rates. Normal, and other distributions will be applied later
    r_light = 1.6
    r_med = 2.3
    r_heavy = 5.8
    r_sed = 0.02
    r_dist = 1.4

    #Averages
    walk_ave = 2

    #Data corresponding to steps and distance walked
    distance = abs(np.random.uniform(walk_ave,4))
    steps = distance*(np.random.randint(1300,1600))

    #Data corresponding to time spent on activity
    t_light = abs(np.random.normal(150, 200))
    t_med = (abs(np.random.uniform(0.1,0.5)))*t_light
    t_heavy = (abs(np.random.uniform(0,0.9)))*t_med
    t_sed = abs(np.random.normal(14000, 6000))
    t_far_burn = t_heavy*(abs(np.random.uniform(0.5,1.0)))

    #Data corresponding to sleep
    t_asleep = abs(np.random.uniform(6.0,7.5))*60
    t_rem = (abs(np.random.uniform(0.10,0.3)))*t_asleep

    #Resting Heart Rate
    rhr = (np.random.randint(70,90))

    Activity_cals = (t_light)*(r_light)+(t_med)*(r_med)+(t_heavy)*(r_heavy) + (t_sed)*(r_sed)

    Cals_burned = Activity_cals + distance*(r_dist) + np.random.randint(100,200)

    Cals_BMR = (abs(np.random.uniform(0.5,0.85)))*(Cals_burned)

    big_boy.loc[each] = pd.Series({ 'Calories Burned':Cals_burned,
 'Calories BMR':Cals_BMR,
 'Steps':steps,
 'Distance (Km)': distance,
 'Resting Heart Rate':rhr,
 'Minutes Sedentary':t_sed,
 'Minutes Lightly Active':t_light,
 'Minutes Fairly Active':t_med,
 'Minutes Very Active':t_heavy,
 'Activity Calories':Activity_cals,
 'Fat Burn minutes':t_far_burn,
 'Minutes Asleep':t_asleep,
 'Minutes REM sleep':t_rem})


for each in patient_list_4:

    #Average calorie burn rates. Normal, and other distributions will be applied later
    r_light = 3
    r_med = 4
    r_heavy = 9
    r_sed = 0.1
    r_dist = 0.5

    #Averages
    walk_ave = 1


    #Data corresponding to steps and distance walked
    distance = abs(np.random.uniform(walk_ave,4))
    steps = distance*(np.random.randint(1300,1600))

    #Data corresponding to time spent on activity
    t_light = abs(np.random.normal(150, 50))
    t_med = (abs(np.random.uniform(0.1,0.5)))*t_light
    t_heavy = (abs(np.random.uniform(0,0.9)))*t_med
    t_sed = abs(np.random.normal(10000, 2500))
    t_far_burn = t_heavy*(abs(np.random.uniform(0.5,1.0)))

    #Data corresponding to sleep
    t_asleep = abs(np.random.uniform(6.0,7.5))*60
    t_rem = (abs(np.random.uniform(0.10,0.3)))*t_asleep

    #Resting Heart Rate
    rhr = (np.random.randint(70,90))

    Activity_cals = (t_light)*(r_light)+(t_med)*(r_med)+(t_heavy)*(r_heavy) + (t_sed)*(r_sed)

    Cals_burned = Activity_cals + distance*(r_dist) + np.random.randint(100,200)

    Cals_BMR = (abs(np.random.uniform(0.5,0.85)))*(Cals_burned)

    big_boy.loc[each] = pd.Series({ 'Calories Burned':Cals_burned,
 'Calories BMR':Cals_BMR,
 'Steps':steps,
 'Distance (Km)': distance,
 'Resting Heart Rate':rhr,
 'Minutes Sedentary':t_sed,
 'Minutes Lightly Active':t_light,
 'Minutes Fairly Active':t_med,
 'Minutes Very Active':t_heavy,
 'Activity Calories':Activity_cals,
 'Fat Burn minutes':t_far_burn,
 'Minutes Asleep':t_asleep,
 'Minutes REM sleep':t_rem})


for each in patient_list_5:

    #Average calorie burn rates. Normal, and other distributions will be applied later
    r_light = 0.5
    r_med = 0.5
    r_heavy = 3
    r_sed = 0.02
    r_dist = 0.5

    #Averages
    walk_ave = 4


    #Data corresponding to steps and distance walked
    distance = abs(np.random.uniform(walk_ave,6))
    steps = distance*(np.random.randint(800,1200))

    #Data corresponding to time spent on activity
    t_light = abs(np.random.normal(200, 100))
    t_med = (abs(np.random.uniform(0.05,0.3)))*t_light
    t_heavy = (abs(np.random.uniform(0,0.7)))*t_med
    t_sed = abs(np.random.normal(7000, 2400))
    t_far_burn = t_heavy*(abs(np.random.uniform(0.3,1.0)))

    #Data corresponding to sleep
    t_asleep = abs(np.random.uniform(7.0,9.5))*60
    t_rem = (abs(np.random.uniform(0.10,0.3)))*t_asleep

    #Resting Heart Rate
    rhr = (np.random.randint(70,80))

    Activity_cals = (t_light)*(r_light)+(t_med)*(r_med)+(t_heavy)*(r_heavy) + (t_sed)*(r_sed)

    Cals_burned = Activity_cals + distance*(r_dist) + np.random.randint(100,200)

    Cals_BMR = (abs(np.random.uniform(0.7,0.95)))*(Cals_burned)

    big_boy.loc[each] = pd.Series({ 'Calories Burned':Cals_burned,
 'Calories BMR':Cals_BMR,
 'Steps':steps,
 'Distance (Km)': distance,
 'Resting Heart Rate':rhr,
 'Minutes Sedentary':t_sed,
 'Minutes Lightly Active':t_light,
 'Minutes Fairly Active':t_med,
 'Minutes Very Active':t_heavy,
 'Activity Calories':Activity_cals,
 'Fat Burn minutes':t_far_burn,
 'Minutes Asleep':t_asleep,
 'Minutes REM sleep':t_rem})



for each in patient_list_6:

    #Average calorie burn rates. Normal, and other distributions will be applied later
    r_light = 1.90
    r_med = 3.2
    r_heavy = 7.8
    r_sed = 0.52
    r_dist = 2

    #Averages
    walk_ave = 4


    #Data corresponding to steps and distance walked
    distance = abs(np.random.uniform(walk_ave,5.5))
    steps = distance*(np.random.randint(1300,1600))

    #Data corresponding to time spent on activity
    t_light = abs(np.random.normal(150, 100))
    t_med = (abs(np.random.uniform(0.1,0.5)))*t_light
    t_heavy = (abs(np.random.uniform(0,0.9)))*t_med
    t_sed = abs(np.random.normal(2300, 1000))
    t_far_burn = t_heavy*(abs(np.random.uniform(0.5,1.0)))

    #Data corresponding to sleep
    t_asleep = abs(np.random.uniform(6.0,7.5))*60
    t_rem = (abs(np.random.uniform(0.10,0.3)))*t_asleep

    #Resting Heart Rate
    rhr = (np.random.randint(70,90))

    Activity_cals = (t_light)*(r_light)+(t_med)*(r_med)+(t_heavy)*(r_heavy) + (t_sed)*(r_sed)

    Cals_burned = Activity_cals + distance*(r_dist) + np.random.randint(100,200)

    Cals_BMR = (abs(np.random.uniform(0.5,0.85)))*(Cals_burned)

    big_boy.loc[each] = pd.Series({ 'Calories Burned':Cals_burned,
 'Calories BMR':Cals_BMR,
 'Steps':steps,
 'Distance (Km)': distance,
 'Resting Heart Rate':rhr,
 'Minutes Sedentary':t_sed,
 'Minutes Lightly Active':t_light,
 'Minutes Fairly Active':t_med,
 'Minutes Very Active':t_heavy,
 'Activity Calories':Activity_cals,
 'Fat Burn minutes':t_far_burn,
 'Minutes Asleep':t_asleep,
 'Minutes REM sleep':t_rem})

big_boy.dropna(inplace=True)
scaled_data = StandardScaler().fit_transform(big_boy.T)

# Run The PCA
pca = PCA(n_components=3)
pca.fit(big_boy)

# Store results of PCA in a data frame
result=pd.DataFrame(pca.transform(big_boy), columns=['PCA%i' % i for i in range(3)])

# Plot initialisation
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], cmap="Set2_r")

# make simple, bare axis lines through space:
# xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0, 0), (0,0))
# yAxisLine = ((0, 0), (min(result['PCA1']), max(result['PCA1'])), (0,0))
# zAxisLine = ((0, 0), (0,0), (min(result['PCA2']), max(result['PCA2'])))

# label the axes
# ax.set_xlabel("PC1")
# ax.set_ylabel("PC2")
# ax.set_zlabel("PC3")
#plt.show()

n_components = np.arange(1, 10)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(result.values)
          for n in n_components]

plt.plot(n_components, [m.bic(result.values) for m in models], color='red', marker = 'x',markersize=10, linewidth=2, alpha = 0.9)
plt.legend(loc='best')
plt.xlabel('Number of Clusters')
plt.ylabel('BIC Evaluation')

gmm = GaussianMixture(n_components=5).fit(result.values)
labels = gmm.predict(result.values)

df_m = pd.DataFrame(gmm.means_)
print(df_m)

# Plot initialisation
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

# ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], cmap="Set2_r", color='Red')

arrayToPlot = []

for i in range(result.__len__()):
    arrayToPlot.append([0,0,0,0])
    temp = 100000.0
    for j in range(df_m.__len__()):
        distance = np.abs(result['PCA0'][i]-df_m.iloc[j,0]) + np.abs(result['PCA1'][i]-df_m.iloc[j,1]) + np.abs(result['PCA2'][i]-df_m.iloc[j,2])
        if (distance < temp):
            arrayToPlot[i][0] = result['PCA0'][i]
            arrayToPlot[i][1] = result['PCA1'][i]
            arrayToPlot[i][2] = result['PCA2'][i]
            arrayToPlot[i][3] = j
            temp = distance

greenPlot = []
redPlot = []
bluePlot = []
yellowPlot = []
pinkPlot = []

for k in range(arrayToPlot.__len__()):
    print(arrayToPlot[k][3])
    if arrayToPlot[k][3] == 0:
        greenPlot.append([arrayToPlot[k][0],arrayToPlot[k][1],arrayToPlot[k][2]])
    elif arrayToPlot[k][3] == 1:
        redPlot.append([arrayToPlot[k][0],arrayToPlot[k][1],arrayToPlot[k][2]])
    elif arrayToPlot[k][3] == 2:
        bluePlot.append([arrayToPlot[k][0],arrayToPlot[k][1],arrayToPlot[k][2]])
    elif arrayToPlot[k][3] == 3:
        yellowPlot.append([arrayToPlot[k][0],arrayToPlot[k][1],arrayToPlot[k][2]])
    else:
        pinkPlot.append([arrayToPlot[k][0],arrayToPlot[k][1],arrayToPlot[k][2]])


for a in greenPlot:
    ax.scatter(a[0],a[1],a[2], cmap="Set2_r", color='#06785e')
for a in redPlot:
    ax.scatter(a[0],a[1],a[2], cmap="Set2_r", color='#d31d0d')
for a in bluePlot:
    ax.scatter(a[0],a[1],a[2], cmap="Set2_r", color='#00487a')
for a in yellowPlot:
    ax.scatter(a[0],a[1],a[2], cmap="Set2_r", color='#f9aa00')
for a in pinkPlot:
    ax.scatter(a[0],a[1],a[2], cmap="Set2_r", color='#ff798a')

ax.scatter(df_m.iloc[:,0], df_m.iloc[:,1], df_m.iloc[:,2], s=800, marker='+')

# label the axes
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.show()
