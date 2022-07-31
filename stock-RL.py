import sys
import pickle
import numpy as np
sys.getdefaultencoding()
np.set_printoptions(threshold=1000000000000000)
path = 'D:/电子课本习题/暑假实习/one-pixel-attack-keras/networks/results/targeted_results.pkl'
file = open(path,'rb')
inf = pickle.load(file,encoding='iso-8859-1')       #读取pkl文件的内容
print(inf)
#fr.close()
inf=str(inf)
obj_path = 'D:/电子课本习题/暑假实习/2.txt'
ft = open(obj_path, 'w')
ft.write(inf)