import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay, confusion_matrix,precision_recall_curve,precision_recall_fscore_support, classification_report
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder,MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

h = 0.02  # step size in the mesh

plt.style.use('ggplot')

data_white = "winequality-white.csv"
data_red = "winequality-red.csv"

df = pd.read_csv(data_red, sep=';')       # DATA LOAD

df.hist(bins=25,figsize=(10,10))            # DATA HISTOGRAM

X = df.drop('quality', axis=1)              # DATA QUALITY
y = df['quality']                           # DATA QUALITY ONLY

# DATA TRAIN TEST SPLIT 30%-70% 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=40)

norm = MinMaxScaler()               # DATA NORMALIZE
norm_fit=norm.fit(X_train)
new_Xtrain = norm_fit.transform(X_train)
new_Xtest = norm_fit.transform(X_test)

classifier_RandomForest = RandomForestClassifier()       # CLASSIFIER
clf_RandomForest_Fit = classifier_RandomForest.fit(X_train, y_train)      # CLASSIFIER'DA CALISTIR
clf_RandomForest_Score = classifier_RandomForest.score(new_Xtest, y_test)     # CLASSIFIER'DA BASARI SCORE

classifier_KNeighbors = KNeighborsClassifier(n_neighbors=5)
clf_KNeighbors_Fit = classifier_KNeighbors.fit(X_train, y_train)
clf_KNeighbors_Score = classifier_KNeighbors.score(new_Xtest, y_test)

classifier_DecisionTree = DecisionTreeClassifier()
clf_DecisionTree_Fit = classifier_DecisionTree.fit(X_train, y_train),
clf_DecisionTree_Score = classifier_DecisionTree.score(new_Xtest, y_test)

classifier_SVC = SVC()
clf_SVC_Fit = classifier_SVC.fit(X_train, y_train)
clf_SVC_Score = classifier_SVC.score(new_Xtest, y_test)



pred_RandomForest = list(classifier_RandomForest.predict(X_test))     # CLASSIFIER'DA PREDICT
pred_RandomForest_df = {"predicted: ": pred_RandomForest, "actual: ": y_test}
y_test_list = list(y_test)                  # ACTUAL DEĞERLER LİSTESİ

pred_KNeighbors = list(classifier_KNeighbors.predict(X_test))

clf_RandomForest_report = classification_report(pred_RandomForest,y_test)   # CLASSIFIER REPORT
clf_KNeighbors_report = classification_report(pred_KNeighbors,y_test)

conf_matrix_RandomForest = confusion_matrix(y_test, pred_RandomForest)   # CONFUSION MATRIX
conf_matrix_KNeighbors = confusion_matrix(y_test, pred_KNeighbors)


c_arr = [x for x in range(100)]

print("********************************")
print("********************************")
print("                                ")
print("{} Accuracy on test set: {}%".format(classifier_RandomForest,clf_RandomForest_Score*100))
print('{} Accuracy on test set: {}%'.format(classifier_KNeighbors,clf_KNeighbors_Score*100))
print('{} Accuracy on test set: {}%'.format(classifier_DecisionTree,clf_DecisionTree_Score*100))
print('{} Accuracy on test set: {}%'.format(classifier_SVC,clf_SVC_Score*100))
print("                                ")
print("********************************")
print(" RESULTS FOR RANDOM FOREST ONLY ")
print("********************************")
print("                                ")
print("RandomForest Confusion matrix:", conf_matrix_RandomForest)
print('Kneighbors Confusion matrix:', conf_matrix_KNeighbors)
print("                                ")
print("                                ")
print("RandomForest Classification report:", clf_RandomForest_report)
print('KNeighbors Classification report:', clf_KNeighbors_report)
print("                                ")
print("********************************")
print("********************************")
print("                                ")
print(pd.DataFrame(pred_RandomForest_df).head(20))
print("                                ")
print("********************************")


fig, axis = plt.subplots(2,2)

axis[0,0].scatter(y_test_list,pred_RandomForest, alpha=0.15)
axis[0,0].set_xlabel("actual")
axis[0,0].set_ylabel("predicted")
axis[0,0].set_title("Predicted vs Actual")

axis[0,1].plot(c_arr,pred_RandomForest[0:100], c="red")
axis[0,1].plot(c_arr,y_test_list[0:100], c="black")
axis[0,1].set_xlabel("number of samples")
axis[0,1].set_ylabel("quality pred diff")

axis[1,0].hist(pred_RandomForest, bins=10,color="blue", label="predicted")
axis[1,0].hist(y_test_list, bins=10,color="orange",alpha=0.5, label="actual")
axis[1,0].legend()

axis[1,1].bar(["RandomForest"],clf_RandomForest_Score*100, color="black")
axis[1,1].bar(["KNeighbors"],clf_KNeighbors_Score*100, color="red")
axis[1,1].bar(["DecisionTree"],clf_DecisionTree_Score*100, color="green")
axis[1,1].bar(["SVC"],clf_SVC_Score*100, color="blue")
axis[1,1].set_ylabel("Accuracy %")

plt.show()
