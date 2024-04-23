import glob
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np


map_path = "/home/mmc_ubuntu/Work/system-infra/Simulation/src/maps/KIAPI_CONTROL_DEMO/"
matfiles = ["waypoint_1.mat",
            "waypoint_2.mat",
            "waypoint_3.mat"
            ]

offset = [6, 0, 0, 1160]

offset = [0, 0, 5, 0]

backup_num = 150

color = ['r--','g--','b--','k--']
for i, matfile in enumerate(matfiles):
    mat = sio.loadmat(map_path+matfile)
    easts = mat["east"][0]
    norths = mat["north"][0]
    stations = mat["station"][0]
    
    new_easts = [easts[0]]
    new_norths = [norths[0]]
    new_stations = [stations[0]]
    
    print(len(easts))
    easts = np.concatenate([easts[-backup_num:], easts[:-backup_num]])
    
    print(len(easts))
    
    norths = np.concatenate([norths[-backup_num:], norths[:-backup_num]])
    stations_st = stations[-backup_num]    
    stations[-backup_num:] = stations[-backup_num:]-stations_st
    stations[:-backup_num] = stations[:-backup_num:]+stations[-1]
    
    stations = np.concatenate([stations[-backup_num:], stations[:-backup_num]])
    
    
    
    
    for j in range(len(easts)-1):
        if easts[j+1]-easts[j] == 0 and norths[j+1]-norths[j] == 0:
            print("herere", [matfile, j ])
        else:
            new_easts.append(easts[j+1])
            new_norths.append(norths[j+1])
            new_stations.append(stations[j+1])
            
            

    # if i==2:
    #     idx = np.where(stations+offset[i]>460)[0] #174
    #     print(idx)
        
    #     idx = np.where(stations+offset[i]>1160)[0] #439
    #     print(idx)
        
        
    #     idx = np.where(stations+offset[i]>1390)[0] #548
    #     print(idx)
    
    print(np.min(easts),np.min(norths))   
    if i==0:
        mat = {"east" : easts[0:200], "north" : norths[0:200], "station" : stations[0:200]}
        
        sio.savemat(map_path+"waypoints_2_rev.mat",mat)
        
        mat = {"east" : easts[500:700], "north" : norths[500:700], "station" : stations[500:700]}
        
        # sio.savemat(map_path+"waypoints_3_rev.mat",mat)
        
       
        
        plt.plot(easts[0:200], norths[0:200], color[i])
        # print(np.min(easts[0:200]),np.min(norths[0:200]))   
        plt.plot(easts[500:700], norths[500:700], color[i])
        print(np.min(easts[500:700]),np.min(norths[500:700]))   
        
        # print(stations[600])
    
        # print(stations[:200])
        
    else:
        # plt.plot(stations)
        
        print(np.min(easts),np.min(norths))   
        if i==1:
            mat = {"east" : easts, "north" : norths, "station" : stations}
            
            sio.savemat(map_path+"waypoints_1_rev.mat",mat)
            
        if i==2:
            mat = {"east" : easts, "north" : norths, "station" : stations}
            
            sio.savemat(map_path+"waypoints_0_rev.mat",mat)
            
        plt.plot(easts, norths, color[i])
        
    plt.plot(easts[0], norths[0], 'ro', markersize=4)
    plt.plot(easts[10], norths[10], 'bo', markersize=4)
    
    
    print("[%s] station[0] = %.2f, stations[-1] = %.2f" % (matfile, stations[0]+offset[i], stations[-1]+offset[i]))
    # print("stations : ", stations[0:100])
    mat = {"east" : new_easts, "north" : new_norths, "station" : new_stations}
    
    # sio.savemat(map_path+matfile[:-4]+"_rev.mat",mat)
plt.axis("equal")
plt.show()
