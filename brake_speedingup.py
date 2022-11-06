import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA

def timestamp2date(timestamp):
    """Convert timestamp to datetime."""
    timeArray = time.localtime(timestamp)
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return formatted_time
def get_tgAcceleration(df):
    """Calculate tangential acceleration."""
    dVehSpdLgtA = df["VehSpdLgtA"].diff() / 3.6
    dt = df["t"].diff()
    df["tg_acceleration"] = dVehSpdLgtA / dt
    return df
def kde2D(x, y, bandwidth, xbins=10j, ybins=10j, **kwargs): 
    """Build 2D kernel density estimate (KDE)."""
    
    # create grid of sample locations (default: 10x10)
    xx, yy = np.mgrid[x.min():x.max():xbins, 
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    zz = np.reshape(z, xx.shape)
    z_norm = z / z.sum()
    z_norm= z_norm.ravel()
    return xx,yy,zz,z_norm

# read data
df = pd.read_csv(r"/Users/liuyihang/Desktop/brakedata.csv")

# convert timestamp
df["date"] = df["t"].apply(timestamp2date)

# convert m/s to km/h
df["VehSpdLgtA"] = df["VehSpdLgtA"] * 3.6
df = df[["t","date","VehSpdLgtA","BrkPedlPsdBrkPedlPsd"]]


# drop reduplicative records
df = df.drop_duplicates(subset = ["t"]).sort_values(by = ["t"]).reset_index(drop = True)

# partition journey
df["dt"] = df["t"].diff()
df["stop"] = df["dt"] != 1
df["journeyID"] = df["stop"].cumsum()
df.loc[df["dt"] != 1].head(5)

# interpolate speed
df["VehSpdLgtA"] = df.groupby("journeyID")["VehSpdLgtA"].apply(lambda v: v.interpolate(limit_direction = "both"))
df["VehSpdLgtA"] = df["VehSpdLgtA"].fillna(0)
df.isnull().sum()

# calculate tangential acceleration
df = df.groupby("journeyID").apply(get_tgAcceleration)

# omit the first row of every journey (beacause of NaN of tg_acceleration)
df = df.groupby("journeyID").apply(lambda x:x.iloc[1:]).reset_index(drop = True)

# omit meaningless journeies(v == 0 or cumsum(t) less than 3min)
df["is_meaningless"] = df.groupby("journeyID")["VehSpdLgtA"].transform(lambda v: True if (np.sum(v) == 0 or len(v) < 3 * 60) else False)
df = df[~df["is_meaningless"]].reset_index(drop = True)
df = df[["t","date","journeyID","VehSpdLgtA","tg_acceleration","BrkPedlPsdBrkPedlPsd"]]


df['suddenbrake'] = 0
df.loc[(df['BrkPedlPsdBrkPedlPsd']=='NoYes1_Yes') & (df['tg_acceleration']<-1.47 ), 'suddenbrake'] = 1

df['suddenbrakesec'] = 0

if df.loc[0,'suddenbrake'] == 1:
    df.loc[0,'suddenbrakesec'] = 1

for i in range(1,len(df)):
    if  df.loc[i,'suddenbrake'] == 1:
        df.loc[i,'suddenbrakesec'] = df.loc[i-1,'suddenbrakesec'] + 1
    if df.loc[i-1,'suddenbrake'] == 1 and df.loc[i,'suddenbrake'] == 0:
        df.loc[i,'suddenbrakesec'] = 0
        
df['suddenbrakeseconds'] = 0
for i in range(0,len(df)-1):      
    if  df.loc[i,'suddenbrake'] == 1 and df.loc[i+1,'suddenbrake'] == 0:
        df.loc[i,'suddenbrakeseconds']= df.loc[i,'suddenbrakesec']

df_brake = df[df['suddenbrakeseconds'] !=0 ]
df_brake = df_brake.reset_index(drop=True)
df_brake = df_brake[["t","date","journeyID","VehSpdLgtA","tg_acceleration","suddenbrakeseconds"]]


df_brake['st'] = df_brake['t']- df_brake['suddenbrakeseconds']+1
df_brake["start"] = df_brake["st"].apply(timestamp2date)
df_brake['suddenbrakeID'] = 1
for i in range(1,len(df_brake)): 
    if  df_brake.loc[i,'journeyID'] == df_brake.loc[i-1,'journeyID']:
        df_brake.loc[i,"suddenbrakeID"] = df_brake.loc[i-1,"suddenbrakeID"] + 1
        
df_brake['end'] = df_brake['date']
df_brake = df_brake[["journeyID","suddenbrakeID","start","end","suddenbrakeseconds","VehSpdLgtA","tg_acceleration"]]


df['speedingup'] = 0
df.loc[df['tg_acceleration']>1.47 , 'speedingup'] = 1

df['speedingupsec'] = 0

if df.loc[0,'speedingup'] == 1:
    df.loc[0,'speedingsec'] = 1

for i in range(1,len(df)):
    if  df.loc[i,'speedingup'] == 1:
        df.loc[i,'speedingupsec'] = df.loc[i-1,'speedingupsec'] + 1
    if df.loc[i-1,'speedingup'] == 1 and df.loc[i,'speedingup'] == 0:
        df.loc[i,'speedingupsec'] = 0
        

df['speedingupseconds'] = 0
for i in range(0,len(df)-1):      
    if  df.loc[i,'speedingup'] == 1 and df.loc[i+1,'speedingup'] == 0:
        df.loc[i,'speedingupseconds']= df.loc[i,'speedingupsec']

df_speedingup = df[df['speedingupseconds'] !=0 ]
df_speedingup= df_speedingup.reset_index(drop=True)
df_speedingup = df_speedingup[["t","date","journeyID","VehSpdLgtA","tg_acceleration","speedingupseconds"]]


df_speedingup['st'] = df_speedingup['t']- df_speedingup['speedingupseconds']+1
df_speedingup["start"] = df_speedingup["st"].apply(timestamp2date)

df_speedingup['speedingupID'] = 1
for i in range(1,len(df_speedingup)): 
    if  df_speedingup.loc[i,'journeyID'] == df_speedingup.loc[i-1,'journeyID']:
        df_speedingup.loc[i,"speedingupID"] = df_speedingup.loc[i-1,"speedingupID"] + 1

df_speedingup['end'] = df_speedingup['date']
df_speedingup = df_speedingup[["journeyID","speedingupID","start","end","speedingupseconds","VehSpdLgtA","tg_acceleration"]]
