from nltk.sentiment.vader import SentimentIntensityAnalyzer
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import statistics as stat
import pandas as pd
import os
import glob
import numpy as np

def article_to_sentiment(file_name):
    analyzer = SentimentIntensityAnalyzer() #initialize analyzer and assign to variable 'analyzer'
    my_file = open(file_name) #open the specified file and assign to variable 'my_file'
    r = my_file.readlines() #read the lines of 'my_file' and save as variable 'r' (a list of strings)

    a = [] #initialize the empty list 'a', where we will store the polarity scores of the individual lines

    #loop through the lines of list 'r' and perform sentiment analysis, save values to new list 'a'
    for i in range(0,len(r)):
        a.append(str(analyzer.polarity_scores(r[i])))

    letter_list = [] #initialize the list where we will store all the letters of the list of polarity scores

    #loop through the list of polarity scores and turn the whole thing into one long string called 'my_string'
    for j in range(0,len(a)):
        for k in range(0,len(a[j])):
            letter_list.append((a[j][k]))
    my_string = ''.join(map(str, letter_list))

    #remove some punctuation from 'my_string', leaving } to be used to split into a list later
    my_string = my_string.replace("'", '')
    my_string = my_string.replace("{",'')
    my_string = my_string.replace(",",'')
    my_string = my_string.replace('  ',' ')
    my_string = my_string.replace(': ', ':')

    #split back into a list of strings with punctuation removed
    my_list = my_string.split("}")

    #initialize my lists of values for the four sentiments, neg, neu, pos, and comp
    neg = []
    neu = []
    pos = []
    comp = []

    #scrapes 'my_list' for the values that correspond to each of the sentiments
    #and sorts them into their respective lists.
    for g in range (0,len(my_list)):
        for h in range(0,len(my_list[g])):
            if (my_list[g][h] == ".") and (my_list[g][h-5:h-1] == "neg:"):
                neg.append(my_list[g][h-1:h+3])
            if (my_list[g][h] == ".") and (my_list[g][h-5:h-1] == "neu:"):
                neu.append(my_list[g][h-1:h+3])
            if (my_list[g][h] == ".") and (my_list[g][h-5:h-1] == "pos:"):
                pos.append(my_list[g][h-1:h+3])
            if (my_list[g][h] == ".") and (my_list[g][h-5:h-1] == "und:"):
                comp.append(my_list[g][h-1:h+3])
            if (my_list[g][h-2] == '-'):
                comp.append(my_list[g][h-2:h+3])

    #initialize a new group of lists, which will store the values of neg, neu, pos,
    #after their values are tranformed to floats
    neg_float = []
    neu_float = []
    pos_float = []
    comp_float = []
    index = []

    #creates an index
    for i in range(0,7211):
        index.append(i+1)

    #scrapes the respective lists, converts them to floats, deposits them
    #into their respective _float lists.
    for eins in range(0,len(neg)):
        neg_float.append(float(neg[eins]))
    for zwei in range(0,len(neu)):
        neu_float.append(float(neu[zwei]))
    for drei in range(0,len(pos)):
        pos_float.append(float(pos[drei]))
    for vier in range(0,len(comp)):
        comp_float.append(float(comp[vier]))

    #initialzes a new list which will only include from instances where
    #comp_float i != 0.0
    neg_float_new = []
    neu_float_new = []
    pos_float_new = []
    comp_float_new = []
    index_new = []

    #create an index
    for i in range(0,7211):
        index_new.append(i+1)

    #scrape comp_float looking for 0.0 values. if this index value has no
    #corresponding comp_float value, remove corresponding neg,neu,float vals
    for i in range(0,len(comp_float)):
        if (comp_float[i] == 0.0):
            pass
        else:
            neg_float_new.append(neg_float[i])
            neu_float_new.append(neu_float[i])
            pos_float_new.append(pos_float[i])
            comp_float_new.append(comp_float[i])

    #calculates the mean of each list, rounding the results to 3 decimal places
    neg = round(stat.mean(neg_float_new),3)
    neu = round(stat.mean(neu_float_new),3)
    pos = round(stat.mean(pos_float_new),3)
    comp = round(stat.mean(comp_float_new),3)
    x = pos
    y = neu
    z = neg
    my_list = [x,y,z,comp]

    return my_list


def scanfolder():
    list_of_paths=[]
    for path, dirs, files in os.walk('/home/jgadasu1/FinalProject-VaderML'):
        for f in files:
            if f.endswith('.txt'):
                list_of_paths.append(os.path.join(path, f))
    articles=[]
    for i in list_of_paths:
         listing = article_to_sentiment(i)+[os.path.basename(os.path.abspath(os.path.join(i, os.pardir)))]
         articles.append(listing)
         frame = pd.DataFrame(articles)
         Title = ['Positive', 'Neutral','Negative','Compound','NewsSource']
         frame.columns = Title
    print (frame)
    frame.to_csv('Frame.csv')
    return frame

def compmodel_ML():

    dataset = pd.read_csv('Frame.csv')
    dataset.drop(['Unnamed: 0'],axis = 1)
    X = dataset.iloc[:, :-1].values
    print (X)
    y = dataset.iloc[:, 5].values
    print (y)


    # Encoding categorical data
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)
    print (y)
    y.reshape(-1,1)
    onehotencoder = OneHotEncoder(categorical_features = [9])
    y = onehotencoder.fit_transform(y).toarray()
    # Avoiding the Dummy Variable Trap
    X = X[:, 1:]
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)




if __name__=='__main__':
    scanfolder()
