import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import pi,cos,atan,sin
from datetime import datetime,timedelta,date
from collections import Counter
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d

data_path = 'D:\McGill Courses\Robust Optimization\Project\datapreprocessing\data\\202106-citibike-tripdata.csv'
downloaddata = pd.read_csv(data_path)
downloaddata['datetime'] = pd.to_datetime(downloaddata['started_at'])
sorted_data = downloaddata.sort_values(by='datetime').reset_index(drop=True)
select_col = ['ride_id','datetime','start_station_id','start_lat','start_lng']
data_df = sorted_data[select_col]
data_df = data_df.dropna().reset_index(drop=True)

# convert coordinate to distance and align
lat = 40.75
longitude_factor = 111.320*cos(lat/180*pi)
latitude_factor = 110.574
delta_distance = (np.array([40.816835, -73.934743])-np.array([40.732365, -73.996315]))*np.array([latitude_factor,longitude_factor])
theta_rotate = atan(delta_distance[1]/delta_distance[0])
data_df['x_beforealign'] = data_df['start_lng']*longitude_factor
data_df['y_beforealign'] = data_df['start_lat']*latitude_factor
data_df['x'] = data_df['x_beforealign']*cos(theta_rotate) - data_df['y_beforealign']*sin(theta_rotate)
data_df['y'] = data_df['x_beforealign']*sin(theta_rotate) + data_df['y_beforealign']*cos(theta_rotate)

'''
# plot this stations
start_time = datetime(2021,6,1,0,0,0)
end_time = datetime(2021,6,10,0,0,0)
data_firstday = data_df[(data_df['datetime']<end_time)&(data_df['datetime']>start_time)]
plt.figure(figsize=(7,12))
plt.scatter(data = data_firstday, x = 'x', y = 'y', alpha=0.15,c='forestgreen',edgecolors='black',s=30,linewidths=0.3)
plt.xlim((-7645,-7638))
plt.ylim((903,915))
plt.savefig('bike_map_selected.jpg', format='jpg', transparent=False, dpi=300, pad_inches = 0)
plt.show()
'''

# select data in this box (12x7 km)
BBox = (-7645,-7638,903,915)
subset_data_df = data_df[(data_df['x']>=BBox[0])&(data_df['x']<BBox[1])&(data_df['y']>=BBox[2])&(data_df['y']<BBox[3])][['ride_id','datetime','start_station_id','x','y']].reset_index(drop=True)
subset_data_df['x'] -= BBox[0]
subset_data_df['y'] -= BBox[2]
BBox = (0,7,0,12) #(x_min, x_max, y_min, y_max)
station_df = subset_data_df[subset_data_df.duplicated(subset='start_station_id', keep='first')==False][['start_station_id','x','y']].reset_index(drop=True)
station_df['region'] = ((BBox[3]-station_df['y'].to_numpy())//1).astype('int')*7+((station_df['x'].to_numpy())//1).astype('int')
station_df.to_csv('data/station_info.csv',index=False)
subset_data_boxed_df = subset_data_df.merge(station_df[['start_station_id','region']],how='left',on='start_station_id')
subset_data_boxed_df.to_csv('data/boxed-202106-citibike-tripdata.csv',index=False)

# aggregate data
subset_data_boxed_df = pd.read_csv('data/boxed-202106-citibike-tripdata.csv')
subset_data_boxed_df['datetime'] = pd.to_datetime(subset_data_boxed_df['datetime'])
service_region_num = 12*7
Start_Time = datetime(2021,6,1,0,0,0)
End_Time = datetime(2021,7,1,0,0,0)
step = timedelta(hours=1)
start_time = Start_Time
datetime_list = []
region_list = []
demand_list = []
while start_time<End_Time:
    end_time = start_time+step
    print('Time: {}'.format(start_time))
    tmp_step_df = subset_data_boxed_df[(subset_data_boxed_df['datetime']>start_time)&(subset_data_boxed_df['datetime']<=end_time)]
    for region_id in range(service_region_num):
        tmp_step_df_one_region = tmp_step_df[tmp_step_df['region']==region_id]
        demand_list.append(len(tmp_step_df_one_region))
    datetime_list.extend([start_time]*service_region_num)
    region_list.extend(list(range(service_region_num)))
    start_time = end_time
demand_df = pd.DataFrame({'datetime':datetime_list,'region':region_list,'demand':demand_list})
demand_df.to_csv('data/demand-boxed-202106.csv',index=0)

#%%
# read date from 202106~202110
station_df = pd.read_csv('data/station_info.csv')
demand_df = pd.read_csv('data/demand-boxed-202106.csv')
new_demand_df = []
service_region_num = 12*7
months_list = [202107,202108,202109,202110]
for i in range(len(months_list)):
    month_ = months_list[i]
    rawdata = pd.read_csv('rawdata\\{}-citibike-tripdata.csv'.format(month_))
    rawdata['start_station_id'] = pd.to_numeric(rawdata['start_station_id'],errors='coerce')
    rawdata = rawdata.dropna()
    rawdata['datetime'] = pd.to_datetime(rawdata['started_at'])
    tmp_subset_data_boxed_df = rawdata.merge(station_df[['start_station_id','region']],how='inner',on='start_station_id')[['datetime','start_station_id','region']]
    tmp_subset_data_boxed_df = tmp_subset_data_boxed_df.sort_values(by='datetime')
    Start_Time = datetime(2021,7+i,1,0,0,0)
    End_Time = datetime(2021,7+i+1,1,0,0,0)
    step = timedelta(hours=1)
    start_time = Start_Time
    datetime_list = []
    region_list = []
    demand_list = []
    while start_time<End_Time:
        end_time = start_time+step
        if start_time.hour==0:
            print('Time: {}'.format(start_time))
        tmp_step_df = tmp_subset_data_boxed_df[(tmp_subset_data_boxed_df['datetime']>=start_time)&(tmp_subset_data_boxed_df['datetime']<end_time)]
        for region_id in range(service_region_num):
            tmp_step_df_one_region = tmp_step_df[tmp_step_df['region']==region_id]
            demand_list.append(len(tmp_step_df_one_region))
        datetime_list.extend([start_time]*service_region_num)
        region_list.extend(list(range(service_region_num)))
        start_time = end_time
    new_demand_df.append(pd.DataFrame({'datetime':datetime_list,'region':region_list,'demand':demand_list}))
demand_df = demand_df.append(new_demand_df).reset_index(drop=True)
demand_df.to_csv('data/demand-boxed-202106to10.csv')

