import func
import numpy as np


###folderu cu clase cu imagini
rootFolder = 'Images/'

Function=func.Functions()
# # # ##astea rulate o sg data pe folder
#Function.generare_fisier_baza_de_date(rootFolder)
#Function.generare_patch_bd(rootFolder + 'trasaturi_labels.txt')

#Function.generare_dictionar(21)

#Function.incarcare_dictionar()

#Function.codare_caracteristici()
#1#labels
#######################################################
train_labels = []
test_labels = []
all_labels = []
with open(rootFolder + 'trasaturi_labels.txt', 'r') as f:
    for i, line in enumerate(f):
        all_labels.append(int(line.split()[1]))
        #print(i, line)
        if i % 2 ==0:
            train_labels.append(line.split()[1])
            #print(train_labels)
        else:
            test_labels.append(line.split()[1])
#######################################################

        
##citire descriptori salvati prin "generare_descriptori"

caracteristici  = np.load('caracteristici.npy')
print (caracteristici)


###################################
#CLASIFICARE, TESTARE, EVALUARE####
###################################
#MODIFICA AICI CU: KNN, SVM


print('pentru KNN')
Function.antrenare_si_testare_knn(caracteristici, train_labels, test_labels)
print('pentru SVM')
Function.antrenare_si_testare_svm(caracteristici, train_labels, test_labels)


