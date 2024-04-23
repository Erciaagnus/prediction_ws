import glob
import scipy.io as sio
import matplotlib.pyplot as plt

matfile = './merge_in.mat'

mat = sio.loadmat(matfile)
easts = mat["east"][0]
norths = mat["north"][0]
stations = mat["station"][0]

for i in range(1, len(stations)):
    if stations[i-1] == stations[i]:
        print(i)

# plt.plot(easts, norths, '.-')
# print("[%s] station[0] = %.2f, stations[-1] = %.2f" % (matfile, stations[0], stations[-1]))
#
# plt.axis("equal")
# plt.show()
