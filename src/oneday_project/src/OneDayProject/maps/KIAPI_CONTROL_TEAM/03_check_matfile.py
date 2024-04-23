import glob
import scipy.io as sio
import matplotlib.pyplot as plt

map_path = "/home/mmc_ubuntu/Work/system-infra/Simulation/src/maps/KIAPI_CONTROL_TEAM/"
matfiles = ["mainlane_1.mat",
            "mainlane_2.mat",
            "merge_in.mat",
            "merge_out.mat"
            ]

offset = [6, 0, 0, 1160]

color = ['r--','g--','b--','k--']
for i, matfile in enumerate(matfiles):
    mat = sio.loadmat(map_path+matfile)
    easts = mat["east"][0]
    norths = mat["north"][0]
    stations = mat["station"][0]
    
    new_easts = [easts[0]]
    new_norths = [norths[0]]
    new_stations = [stations[0]]
    
    for j in range(len(easts)-1):
        if easts[j+1]-easts[j] == 0 and norths[j+1]-norths[j] == 0:
            print("herere", [matfile, j ])
        else:
            new_easts.append(easts[j+1])
            new_norths.append(norths[j+1])
            new_stations.append(stations[j+1])
            
            

    plt.plot(easts, norths, color[i])
    print("[%s] station[0] = %.2f, stations[-1] = %.2f" % (matfile, stations[0]+offset[i], stations[-1]+offset[i]))
    # print("stations : ", stations[0:100])
    mat = {"east" : new_easts, "north" : new_norths, "station" : new_stations}
    
    # sio.savemat(map_path+matfile[:-4]+"_rev.mat",mat)
plt.axis("equal")
plt.show()
