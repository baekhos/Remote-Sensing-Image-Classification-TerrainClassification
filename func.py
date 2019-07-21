import numpy as np
import cv2, os
import sklearn
from skimage import io,color,util
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import random
class Functions():
    def __init__(self):        
        self.patch_list=[]
        self.dict=[]
        self.index = []
        self.cod = []
        self.hist=[]
        
    def codare_caracteristici(self):
        caracteristici=[]
        patches_in_an_image=7225
        for i in range(0,2100):
            hist=[]
            for x in self.dict:
                a=patches_in_an_image*i
                b=patches_in_an_image*(i+1)
                hist.append(self.patch_list[a:b].count(x))
            caracteristici.append(hist)
        caracteristici_np=np.array(caracteristici)
        np.save('caracteristici',caracteristici_np)
        

    def generare_fisier_baza_de_date(self,rootFolder, debug=0):
        dirs = os.listdir(rootFolder)
        if debug: print(dirs)
        class_no = 0
        with open(rootFolder+'trasaturi_labels.txt', 'w') as f:
            for dir in dirs:
                if os.path.isdir(os.path.join(rootFolder, dir)):
                    class_no += 1
                    for image_path in os.listdir(os.path.join(rootFolder, dir)):
                        one_line = rootFolder + dir + "/" + image_path +" "+ str(class_no)
                        f.write("%s \n" % one_line)
                        
    def generare_patch_bd(self,cale_fisier_trasaturi, debug=0):        
        with open(cale_fisier_trasaturi, 'r') as f:
            for line in f:
                image = io.imread(line.split()[0])
                index= int(line.split()[1])
                image = color.rgb2gray(image)            
                self.generare_patch(image,index)  
    

    def generare_dictionar(self, index):  
        diviziune=int(len(self.patch_list)/index)
#        print (diviziune)
        for i in range(0,index):
            a=diviziune*i
            b=diviziune*(i+1)
            my_randoms = random.sample(range(a, b), 10)
            for n in my_randoms:
                self.dict.append(self.patch_list[n])        
#        
    def incarcare_dictionar(self):
        self.dict=np.load('desc.npy')
                        
                        
    def codarea_caracteristicilor(self):  
        for x in self.patch_list:
            d=np.linalg.norm(np.array(self.desc)-x, axis=1)
            self.cod.append(np.argmin(d))
        print ("Final codare caracteristici")    
    
    def codarea_caracteristicilor_2(self):
#        patch=np.load('patch.npy')
#        print(patch.shape)
#        cod=[]
        lp = len(self.patch_list)
        ld = len(self.desc)
        cod = np.empty([lp,ld]) 
        for i in range(ld):
            x = self.desc[i]
            cod[:,i]=np.linalg.norm(np.array(self.patch_list)-x, axis=1)

    def generare_patch(self,img,index):
        s=img.shape
        for i in range(1,s[0],3):
            for j in range(1,s[1],3):
                try:
                    v=img[i-1:i+2,j-1:j+2].reshape(9)
                    self.patch_list.append(v)
                    self.index.append(index)
                except:
                    pass
        

    
    def antrenare_si_testare_knn(self,trasatauri, train_labels, test_labels, debug=0):
        Train = trasatauri[0:trasatauri.shape[0]:2,:]
        Test  = trasatauri[1:trasatauri.shape[0]:2,:]
    
        print('Avem {} pentru antrenare si {} pentru testare'.format(Train.shape, Test.shape))
    
        clasificator = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree').fit(Train, train_labels)
    
        predict_nb = clasificator.predict(Test)
    
        confusionmatrix_nb = sklearn.metrics.confusion_matrix(test_labels, predict_nb)
        print(' KNN accuracy is {}'.format(sklearn.metrics.accuracy_score(predict_nb,test_labels)))
    
        return sklearn.metrics.accuracy_score(predict_nb,test_labels)
    
    def antrenare_si_testare_svm(self,trasatauri, train_labels, test_labels, debug=0):
        Train = trasatauri[0:trasatauri.shape[0]:2,:]
        Test = trasatauri[1:trasatauri.shape[0]:2,:]
    
        if debug==1:print('Avem {} pentru antrenare si {} pentru testare'.format(Train.shape, Test.shape))
    
        clasificator = svm.LinearSVC().fit(Train, train_labels)
    
        predict_nb = clasificator.predict(Test)
    
        confusionmatrix_nb = sklearn.metrics.confusion_matrix(test_labels, predict_nb)
        print('SVM accuracy is {}'.format(sklearn.metrics.accuracy_score(predict_nb,test_labels)))
    
        return sklearn.metrics.accuracy_score(predict_nb,test_labels)
    
    
    def norm_descriptor(self,descriptor):
        descriptor_min = np.min(descriptor, axis=1).reshape(descriptor.shape[0], 1)
        descriptor_max = np.max(descriptor, axis=1).reshape(descriptor.shape[0], 1)
        descriptor_norm = (descriptor - descriptor_min) / (descriptor_max - descriptor_min)
        return descriptor_norm
    #
    #def z_norm_descriptor(self,descriptor):
    #    descriptor_mean = descriptor.mean(1).reshape(descriptor.shape[0], 1)
    #    descriptor_std = np.std(descriptor, axis=1).reshape(descriptor.shape[0],1)
    #    descriptor_z_norm = (descriptor - descriptor_mean)/descriptor_std
    #    return descriptor_z_norm