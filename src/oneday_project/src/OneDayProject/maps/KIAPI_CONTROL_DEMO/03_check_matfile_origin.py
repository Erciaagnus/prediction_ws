import glob
import scipy.io as sio
import matplotlib.pyplot as plt

# matfiles = glob.glob("./*.mat")
map_path = "/home/mmc_ubuntu/Work/system-infra/Simulation/src/maps/KIAPI_CONTROL_TEAM"
matfiles = ["mainlane_1.mat",
            "mainlane_2.mat",
            "merge_in.mat",
            "merge_out.mat"
            ]
        

color = ['r--','g--','b--','k--']
for i, matfile in enumerate(matfiles):
    mat = sio.loadmat(map_path+matfile)
    easts = mat["east"][0]
    norths = mat["north"][0]
    stations = mat["station"][0]

    plt.plot(easts, norths, color[i])
    print("[%s] station[0] = %.2f, stations[-1] = %.2f" % (matfile, stations[0], stations[-1]))

plt.axis("equal")
plt.show()
