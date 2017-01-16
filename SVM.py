
# coding: utf-8

# In[ ]:

from sklearn.datasets import load_svmlight_file
from sklearn import cross_validation
from sklearn import svm
import numpy as np
from sklearn.grid_search import GridSearchCV


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn import preprocessing

def generate_parsed_data() :
    file_to_read = "scop_motif.data"

    with open(file_to_read) as f:
        content = f.readlines()

    for i in range(len(content)) :
        if content[i].find("rest") != -1 :
    #         print content[i]
            content[i] = content[i].replace(content[i][:(content[i].find("rest") + 4)], "-1.0")
        else :
            content[i] =  content[i].replace(content[i][:(content[i].find("a.1.1.2") + 7)], "1.0")

    # print content

    file_to_updated = "scop_motif_updated.data"

    updated = open(file_to_updated, "w")
    for iterate in range(len(content)) :
        updated.write(content[iterate])


    X, y = load_svmlight_file(file_to_updated)

#     print "Size Of X", X.shape
#     print "Size Of Y", y.shape
    return X, y

def compare_soft_and_kernel_poly(X, y, isNorm) :
    Cs = np.logspace(-3, 5, 9)
    
    degrees_dynamic = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    c_data = []
    degrees_data = []
    accuracy_data = []
    
    folds = cross_validation.StratifiedKFold(y, 5, shuffle=True)
    for changing_C in range(len(Cs)) :
        inner = []
        for changing_degree in range(len(degrees_dynamic)) :
#             for_poly_dynamic = {'C': 0.01, 'degree': 2, 'kernel': ['poly']}
#     print "For_Poly", for_poly_dynamic
    
#     print "X", X
#     print "Y", y
#     cv = cross_validation.StratifiedKFold(y, 5)
            
            classifier_dynamic = svm.SVC(kernel='poly', degree=degrees_dynamic[changing_degree], C=Cs[changing_C])
            print classifier_dynamic
            cross_validation_result_dynamic = cross_validation.cross_val_score(classifier_dynamic, X, y, cv=folds, scoring='roc_auc')
#             print "Results", cross_validation_result_dynamic
            print "Mean Value: ", np.mean(cross_validation_result_dynamic)
    
            inner.append((np.mean(cross_validation_result_dynamic)))
        accuracy_data.append(inner)
        inner = []
     
    c_data = Cs
    degrees_data = degrees_dynamic
#     c_data = np.log10(Cs)
    
    
    c_data, degrees_data = np.meshgrid(c_data, degrees_data)
    accuracy_data = np.asarray(accuracy_data)
    if isNorm == 0 :
        print "Data Is Not Normalized"
    else :
        print "Data Is Normalized"
    print "C_Data", c_data
    print "Degrees_Data", degrees_data
    print "Accuracy_Data", accuracy_data
    
    plot_3D_graph_of_accuracy(c_data, degrees_data, accuracy_data, 0, isNorm)

    
#     classifier_dynamic = svm.SVC(for_poly_dynamic).fit(X, y)
    
#     cross_validation.cross_val_score(classifier_dynamic, X, y, cv=5, scoring='roc_auc')
    
    
def compare_soft_and_kernel_gaussian(X, y, isNorm) :
    Cs = np.logspace(-3, 5, 9)
    
    gammas_dynamic = np.logspace(-5, 3, 9)
    
    folds = cross_validation.StratifiedKFold(y, 5, shuffle=True)
    c_data = []
    gamma_data = []
    accuracy_data = []
#     outer = 0
#     inner = 0
    for changing_C in range(len(Cs)) :
    
        inner = []
        
        for changing_gamma in range(len(gammas_dynamic)) :
#             for_poly_dynamic = {'C': 0.01, 'degree': 2, 'kernel': ['poly']}
#     print "For_Poly", for_poly_dynamic
    
#     print "X", X
#     print "Y", y
#     cv = cross_validation.StratifiedKFold(y, 5)
            
            classifier_dynamic = svm.SVC(kernel='rbf', gamma=gammas_dynamic[changing_gamma], C=Cs[changing_C])
            print classifier_dynamic
            cross_validation_result_dynamic = cross_validation.cross_val_score(classifier_dynamic, X, y, cv=folds, scoring='roc_auc')
#             print "Results", cross_validation_result_dynamic
            print "Mean Value: ", np.mean(cross_validation_result_dynamic)
#             c_data.append(Cs[changing_C])
#             gamma_data.append(gammas_dynamic[changing_gamma])
#             accuracy_data.append(np.mean(cross_validation_result_dynamic))

            inner.append((np.mean(cross_validation_result_dynamic)))
        accuracy_data.append(inner)
        inner = []
        
#     classifier_dynamic = svm.SVC(for_poly_dynamic).fit(X, y)
    
#     cross_validation.cross_val_score(classifier_dynamic, X, y, cv=5, scoring='roc_auc')
    c_data = (Cs)
    gamma_data = (gammas_dynamic)
    
    c_data, gamma_data = np.meshgrid(c_data, gamma_data)
    accuracy_data = np.asarray(accuracy_data)
#     print "X", XNew
#     print "Y", YNew
    print "C_Data", c_data
    print "Gamma_Data", gamma_data
    print "Accuracy_Data", accuracy_data
    if isNorm == 0 :
        print "Data Is Not Normalized"
    else :
        print "Data Is Normalized"
#     c_data = np.asarray(c_data)
#     gamma_data = np.asarray(gamma_data)
#     accuracy_data = np.asarray(accuracy_data)

#     print "C_Shape", (c_data.shape)
#     print "Gamma_Shape", (gamma_data.shape)
#     print "Accuracy_Shape", (accuracy_data.shape)
    
# #     accuracy_data.reshape(gamma_data.size,c_data.size)
    
#     print "C_Data", c_data
#     print "Gamma_Data", gamma_data
#     print "Accuracy_Data", accuracy_data
#     print "Accuracy_Shape", (accuracy_data.shape)
    plot_3D_graph_of_accuracy(c_data, gamma_data, accuracy_data, 1, isNorm)
    
def plot_3D_graph_of_accuracy(x_axis_data, y_axis_data, z_axis_data, flag, norm):
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    X = np.log10(x_axis_data)
    Y = y_axis_data
    if flag == 1:
        Y = np.log10(Y)
    
    Z = z_axis_data
#     print X
#     print Y
#     print Z
    ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
    cset = ax.contour(X, Y, Z, zdir='z', offset=1, cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, zdir='x', offset=np.min(x_axis_data), cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, zdir='y', offset=np.max(y_axis_data), cmap=cm.coolwarm)

    ax.set_xlabel('log(C)')
#     plt.xscale('log')
    title_to_put = 'Title: '
    if flag == 1:
        ax.set_ylabel('Gamma')
        title_to_put = title_to_put + 'Gaussian'
    elif flag == 0:
        ax.set_ylabel('Degrees')
        title_to_put = 'Poly'

#     plt.yscale('log')
    ax.set_zlabel('Accuracy')
    if norm == 0:
        title_to_put = title_to_put + ' & Not Normalized Data'
    else :
        title_to_put = title_to_put + ' & Normalized Data'
    plt.title(title_to_put)
    ax.set_zlim(0, 1)




    plt.show()


    
def nested_cross_validate(X, y, isNorm) :
    Cs = np.logspace(-3, 5, 9)
#     print "Hello"
#     final_param_grid = [{'C': Cs, 'kernel': ['linear']},{'C': Cs, 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
#     final_param_grid = [{'C': Cs, 'kernel': ['linear']}]
#     Final Grid For Gaussian
    gammas = np.logspace(-5, 3, 9)
    for_gaussian = {'C': Cs, 'gamma': gammas, 'kernel': ['rbf']}
    print "For_Gaussian", for_gaussian
    
#     degrees = [2, 3, 4, 5]
#     for_poly = {'C': Cs, 'degree': degrees, 'kernel': ['poly']}
#     print "For_Poly", for_poly
    
#     final_param_grid = [for_poly, for_gaussian]
    final_param_grid = [for_gaussian]
    print "Final_Param", final_param_grid
    folds = cross_validation.StratifiedKFold(y, 5, shuffle=True)
    classifier = GridSearchCV(estimator=svm.SVC(), param_grid=final_param_grid, cv=folds, scoring='roc_auc')
    
    
    classifier.fit(X, y)
    
    
#     cross_validation_results = cross_validation.cross_val_score(classifier, X, y, cv=folds, scoring='roc_auc')
    if isNorm == 0 :
        print "Data Is Not Normalized"
    else :
        print "Data Is Normalized"
#     print "Results_Of_Nested_CV", cross_validation_results
    print classifier.best_score_
    print classifier.best_estimator_
    print classifier.best_params_
    
    
if __name__=='__main__' :
    X, y = generate_parsed_data()
    
    print "Size Of X", X.shape
    print "Size Of Y", y.shape
    
#     nested_cross_validate(X, y)
    #0 For Not Normalized
 
    compare_soft_and_kernel_poly(X, y, 0)
    compare_soft_and_kernel_gaussian(X, y, 0)
    
    print "Before Normalization", X
    X_normalized = preprocessing.normalize(X)
    print "After Normalization", X_normalized
    #1 For Normalized
    compare_soft_and_kernel_gaussian(X_normalized, y, 1)
 
    nested_cross_validate(X, y, 0)
    
    nested_cross_validate(X_normalized, y, 1)
#     X_normalized = []
#     for iterate in range(X.shape[0]) :
#         inner_array = []
#         for inner_iteration in range((X[iterate].shape[0])) :
#             inner_array.append(X[iterate][inner_iteration])
        
# #         print "InnerArray", inner_array
#         X_normalized.append(inner_array)
#         inner_array = []
    
#     print X_normalized
    
# print readFile.read()


# In[ ]:




# In[ ]:



