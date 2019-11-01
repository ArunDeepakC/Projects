#1) Naive Bayes classifier implementation
import scipy.io
import numpy as np
Numpyfile= scipy.io.loadmat('mnist_data.mat')
#Extract X values of training set into numpy array tr_x
tr_x=Numpyfile['trX']
#Extract y labels of training set into numpy array tr_y
tr_y=Numpyfile['trY']
#Extract X values of testing set into numpy array ts_x
ts_x=Numpyfile['tsX']
#Extract y labels of testing set into numpy array ts_y
ts_y=Numpyfile['tsY']
#Feature Extraction from the training set
#Calculate avearge of pixel values of each image
f_mean=list()
for ele in tr_x:
    f_mean.append(np.mean(ele))
f_mean=np.array(f_mean)
#Calculate standard deviation of all pixel values
f_std=list()
for ele in tr_x:
    f_std.append(np.std(ele))
f_std=np.array(f_std)
#Feature Extraction from the test set
#Calculate avearge of pixel values of each image
f_mean_test=list()
for ele in ts_x:
    f_mean_test.append(np.mean(ele))
#Calculate standard deviation of all pixel values
f_std_test=list()
for ele in ts_x:
    f_std_test.append(np.std(ele))
#Calculate prior P(y=7)
def prior_7():
    return 6265/(6265+5851)
#Calculate prior P(y=8)
def prior_8():
    return 5851/(6265+5851)
#Convert array to list
f1_list=f_mean.tolist()
f2_list=f_std.tolist()
label=np.transpose(tr_y).tolist()
#Mean and Std seperated for 7 and 8
f17=list()
f27=list()
f18=list()
f28=list()
for i in range(len(label)):
    if label[i]==[0.0]:
        f17.append(f1_list[i])
        f27.append(f2_list[i])
    if label[i]==[1.0]:
        f18.append(f1_list[i])
        f28.append(f2_list[i])
# Find mu(mean) and sigma(standard deviation) for each feature for given y
F17=np.array(f17)
F27=np.array(f27)
F18=np.array(f18)
F28=np.array(f28)
F17_mean=np.mean(F17)
F27_mean=np.mean(F27)
F18_mean=np.mean(F18)
F28_mean=np.mean(F28)
F17_std=np.std(F17)
F27_std=np.std(F27)
F18_std=np.std(F18)
F28_std=np.std(F28)
# Calculating Posterior Probability P(x1|y) for each case and applying Bayes rule using normal distribution
prob_final=list()
for i in range(len(f_mean_test)): #x_mean,x_std in map(none,f_mean_test,f_std_test):
    p_mean_7 = 1/(np.sqrt(2*np.pi)*F17_std) * np.exp((-(f_mean_test[i]-(F17_mean))**2)/(2*(F17_std)**2))
    p_std_7 = 1/(np.sqrt(2*np.pi)*F27_std) * np.exp((-(f_std_test[i]-(F27_mean))**2)/(2*(F27_std)**2))
    p_mean_8 = 1/(np.sqrt(2*np.pi)*F18_std) * np.exp((-(f_mean_test[i]-(F18_mean))**2)/(2*(F18_std)**2))
    p_std_8 = 1/(np.sqrt(2*np.pi)*F28_std) * np.exp((-(f_std_test[i]-(F28_mean))**2)/(2*(F28_std)**2))
    pof7=p_mean_7*p_std_7*prior_7()
    pof8=p_mean_8*p_std_8*prior_8()
    if pof7>pof8:
        prob_final.append(0.0)
    else:
        prob_final.append(1.0)
testset=list()
for ele in ts_y.tolist():
    testset=ele
# Evaluate accuracy by seeing the number of values the algorithm predicted correct in comparison to test set y
count=0
count_7=0
count_8=0
for i in range(len(testset)):
    if testset[i]==prob_final[i]:
        count=count+1
#Evaluate accuracy of prediction of 7
for i in range(len(testset)):
    if testset[i]==prob_final[i] and testset[i]==0.0:
        count_7=count_7+1
#Evaluate accuracy of prediction of 8
for i in range(len(testset)):
    if testset[i]==prob_final[i] and testset[i]==1.0:
        count_8=count_8+1
accuracy_overall=(count/len(testset))*100
accuracy_7=(count_7/1028)*100
accuracy_8=(count_8/974)*100
#Print the accuracy and the predicted labels
print('NAIVE BAYES')
print('\nNaive Bayes predictions on test set of size ',len(testset), '(Note Label-0.0=>7 Label-1.0=>8):\n',prob_final)
print('\nOverall Accuracy of Naive Bayes on test set:',(count/len(testset))*100,'%')
print('Naive Bayes classification accuracy of digit 7:',(count_7/1028)*100,'%')
print('Naive Bayes classification accuracy of digit 8:',(count_8/974)*100,'%')
print('\n PLEASE WAIT FOR LOGISTIC REGRESSION TO LOAD') 
############################################################################################################################

#2) Logistic Regression classifier
import scipy.io
import numpy as np
Numpyfile= scipy.io.loadmat('mnist_data.mat')
#Extract X values of training set into numpy array tr_x
tr_x=Numpyfile['trX']
#Extract y labels of training set into numpy array tr_y
tr_y=Numpyfile['trY']
#Extract X values of testing set into numpy array ts_x
ts_x=Numpyfile['tsX']
#Extract y labels of testing set into numpy array ts_y
ts_y=Numpyfile['tsY']
#Extracting features from the training set
#Calculate avearge of pixel values of each image
f_mean=list()
for ele in tr_x:
    f_mean.append(np.mean(ele))
f_mean=np.array(f_mean)
#Calculate standard deviation of all pixel values
f_std=list()
for ele in tr_x:
    f_std.append(np.std(ele))
f_std=np.array(f_std)

#Extracting features from the testing set
#Calculate avearge of pixel values of each image
f_mean_test=list()
for ele in ts_x:
    f_mean_test.append(np.mean(ele))
f_mean_test=np.array(f_mean_test)
#Calculate standard deviation of all pixel values
f_std_test=list()
for ele in ts_x:
    f_std_test.append(np.std(ele))
f_std_test=np.array(f_std_test)
#Create train matrix X_train
F_mean=f_mean.tolist()
F_std=f_std.tolist()
X_train=np.column_stack(([1]*len(F_mean),F_mean, F_std))

#Create test matrix X_test
F_mean_test=f_mean_test.tolist()
F_std_test=f_std_test.tolist()
X_test=np.column_stack(([1]*len(F_mean_test),F_mean_test, F_std_test))
#Create a list for y labels in the training set
tr_y.tolist()
new=list()
for ele in tr_y.tolist():
    new=ele
#Create a list for y labels in the testing set
new_test=list()
for ele in ts_y.tolist():
    new_test=ele
#Logistic Regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def gradient(X,y,pofy):
    delta_l= np.dot(X.T,(y-pofy))
    return delta_l
#Training the model and estimating paramters W
def train(X,y,lr,itr):
    no_of_parameter=3
    weight=np.zeros(no_of_parameter)
    for i in range(itr):
        z= np.dot(X, weight)
        pofy= sigmoid(z)
        #Gradient Ascent 
        weight=weight+lr*gradient(X,y,pofy)
    return weight
w=train(X_train,new,7e-4,50000)
#Running the test set in the model
def predict(X_test,weight):
    z= np.dot(X_test, weight)
    pofy= sigmoid(z)
    return pofy
result=predict(X_test,w)
final_classification=list()
for ele in result:
    if ele<0.5:
        final_classification.append(0.0)
    else:
        final_classification.append(1.0)
#Calculating accuracy by comparing y values to 
count=0
count_7_LR=0
count_8_LR=0
y=new_test
for i in range(len(y)):
    if y[i]==final_classification[i]:
        count=count+1
#Evaluate accuracy of prediction of 7
for i in range(len(y)):
    if y[i]==final_classification[i] and y[i]==0.0:
        count_7_LR=count_7_LR+1
#Evaluate accuracy of prediction of 8
for i in range(len(y)):
    if y[i]==final_classification[i] and y[i]==1.0:
        count_8_LR=count_8_LR+1
accuracy_overall_LR=(count/len(y))*100,'%'
accuracy_7_LR=(count_7_LR/1028)*100
accuracy_8_LR=(count_8_LR/974)*100
print('\nLOGISTIC REGRESSION')
print('\nPredicted labels for elements in test set of size',len(y),':\n',final_classification)
print('\nOverall Classification Accuracy of Logistic Regression:',(count/len(y))*100,'%')
print('Logistic Regression classification accuracy of digit 7:',(count_7_LR/1028)*100,'%')
print('Logistic Regression classification accuracy of digit 8:',(count_8_LR/974)*100,'%')
print('\nEstimated values of parameters:[w0,w1,w2]',w)        


