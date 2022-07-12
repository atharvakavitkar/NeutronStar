import pickle
import glob
import numpy as np

data_dir = 'D:/Masters/NS_EoS/dataset/res_nonoise1000x/'
dat_files = glob.glob(data_dir+'*.dat')
for fn,data_file in enumerate(dat_files):
    star_list = []
    lines = open(data_file, "r").readlines()
    with open(data_file) as file:
        lines = file.readlines()
        print(fn,len(lines),'\n',data_file,'\n\n')
        for ln,line in enumerate(lines):
            if 'Star' in line:
                star = {}
            if 'Spectral Coefficients' in line:
                eos = [np.float64(i) for i in lines[ln+1].split()]
                star['m1'] = eos[1]
                star['m2'] = eos[2]
            if 'Mass' in line:
                star_params = [np.float32(i) for i in lines[ln+1].split()]
                star['Mass'] = star_params[0]
                star['Radius'] = star_params[1]
                star['nH'] = star_params[2]
                star['logTeff'] = star_params[3]
                star['dist'] = star_params[4]
            if 'Spectrum' in line:
                spectrum = lines[ln+1].split(",")[:-1]
                for i,s in enumerate(spectrum):
                    star['s'+str(i)] = np.float32(s)
                star_list.append(star)

    filename = 'D:/Masters/NS_EoS/dataset/pkl_files_new/'+data_file.split("\\")[-1].replace('.dat','.pkl')
    with open(filename, 'wb') as handle:
        pickle.dump(star_list, handle, protocol=pickle.HIGHEST_PROTOCOL)



