# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:36:56 2017

@author: David
"""
from __future__ import division
import os
import pandas as pd
import random 
from sklearn import svm
from sklearn.naive_bayes import GaussianNB


def copy_clean_data(_data):
    rating_content = []
    count_P1 = 0
    count_N1 = 0
    length = len(_data)
    for index in range(0,length):
        rating_content.append(_data[index])
        if (rating_content[index] == "40推荐" or rating_content[index] =="50力荐"):
            rating_content[index] = 1
            count_P1 += 1
        elif(rating_content[index] == "10很差" or rating_content[index] == "20较差"):
            rating_content[index] = 0
            count_N1 += 1
        else:
            rating_content[index] = -1
    #print("Totally have ",count_P1," positive data")
    #print("Totally have ",count_N1," negative data")
    return rating_content

def combine_training_data(_comment_data,_rating_data):
    final_data=[]
    negative_number = 0
    if (len(_comment_data) == len(_rating_data)):
        length=len(_comment_data)
        for index in range(0,length):
            if (_rating_data[index] != -1):
                #print("rating ",_rating_data[index])
                #print("comment ", _comment_data[index])
                temp = []
                temp.append(_rating_data[index])
                temp.append(_comment_data[index])
                final_data.append(temp)
    else:
        print ("The length of comment and rating are wrong!")
    return final_data

def divide_data2P_N(_data):
    length = len(_data)
    positive_data = []
    negative_data = []
    for index in range(0,length):
	#print("test data[]:",_data[index][])
	if(_data[index][0]==1):
	    positive_data.append(_data[index])
	else:
	    negative_data.append(_data[index])
    #print("positive data: ",len(positive_data))
    #print("negative data: ",len(negative_data))
    return positive_data, negative_data

def extract_negative_data(_data):
    length = len(_data)
    negative_data = []
    negative_number = 20000
    for index in range(0,length):
	if(negative_number==0):
	    break;	
	if(_data[index][0] == 0):
	    negative_data.append(_data[index])
	    negative_number -= 1
    return negative_data

def extract_positive_data(_data):
    length = len(_data)
    positive_data = []
    positive_number = 20000
    for index in range(0,length):
	if(positive_number==0):
	    break;	
	if(_data[index][0] == 1):
	    positive_data.append(_data[index])
	    positive_number -= 1
    return positive_data

def combine_P_N(_p_data,_n_data):
    length = len(_p_data)
    print ("test: ",length)
    secure_random = random.SystemRandom()
    # according to the number of negative cases is obvisouly samaller than positive cases number, in order to get the same number of positive and negative cases, we use the negative number as baseline.
    final_data = []
    for index in range(0,length):
	final_data.append(_p_data[index])
	final_data.append(secure_random.choice(_n_data))
	#final_data.append(_p_data[index])
    print("Final training data: ",len(final_data))
    return final_data
	

def check_P_N_test(y):
    p_test = 0
    n_test = 0
    for index in range(0,len(y)):
	if(y[index]==1):
	    p_test += 1
	else:
	    n_test += 1
    print("Positive in test data: ",p_test)
    print("Negative in test data: ",n_test)

def correction_predict(y_predict,y):
    count = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for index in xrange(len(y_predict)):
        if(y_predict[index] == y[index]):
            count += 1
	    if(y_predict[index]==1 and y[index] == 1):
	        TP += 1
	    else:
		TN += 1
	else:
	    if(y[index]==1):
		FP += 1
	    else:
		FN += 1
    precision = TP/(TP+FP)*100
    recall = TP/(TP+FN)*100
    accuracy = count/len(y)*100
    f1 = (2*precision*recall)/(precision+recall)
    print("Accuracy: ",accuracy)
    print("Precision: ",precision," %")
    print("Recall: ",recall,"%")
    print("F1: ",f1,"%")

file = open('transfered_Comment_1.csv','rb')
content = file.readlines()
file.close()
for index in range(0,len(content)):
    content[index] = content[index].split(',')
    content[index] = content[index][1::2]

rating_info = pd.read_csv('seg_file1.csv',names=['Movie_Rating','Movie_Comment'])
#print ("Using NuSVM; Nu=0.4")
print ("Using SVM")

rating_info = rating_info['Movie_Rating'].tolist()[1:]
print("test for the first element: ",rating_info[0],'; second: ',rating_info[1])
print("rating info number:", len(rating_info))
print("content number:", len(content))
rating_data = copy_clean_data(rating_info)
print("rating data number:", len(rating_data))
final_data = combine_training_data(content,rating_data)



#positive_data, negative_data = divide_data2P_N(final_data)
#print("The number of final data",len(final_data))
#positive_data = extract_positive_data(final_data)
#negative_data = extract_negative_data(final_data)
#print("The number of positive data",len(positive_data))
#print("The number of negative data",len(negative_data))

#final_data = combine_P_N(positive_data,negative_data)
print("the final_data of input:",len(final_data))

y = []
x = []
for index in range(0,len(final_data)):
    y.append(final_data[index][0])
    x.append(final_data[index][1])
check_P_N_test(y)
print("Totally we have ",len(x)," data.")
training_length = int(round(0.7*len(x)))
print("Training case number: ",training_length)
training_y = y[:training_length]
training_x = x[:training_length]

test_y = y[training_length:]
test_x = x[training_length:]

#print ('Test case number:',len(test_y))
#print ('Training case number:',len(training_x))
check_P_N_test(test_y)


#clf = svm.NuSVC(nu=0.4)
clf = svm.SVC()
clf.fit(training_x,training_y)

predict_y = clf.predict(test_x)
correction_predict(predict_y,test_y)


