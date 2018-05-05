from nltk.sentiment.vader import SentimentIntensityAnalyzer
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import statistics as stat
import pandas as pd
import os
import glob
import numpy as np
import newspaper


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
    neg = stat.mean(neg_float_new)
    neu = stat.mean(neu_float_new)
    pos = stat.mean(pos_float_new)
    comp = stat.mean(comp_float_new)
    x = pos
    y = neu
    z = neg
    my_list = [x,y,z,comp]
    return my_list

def url_to_sentiment(url):
    """Takes a URL from the user, """
    from newspaper import Article
    a = Article(url)
    a.download()
    a.parse()
    article = a.text[:]
    r = str(article)
    r = r.splitlines()
    analyzer = SentimentIntensityAnalyzer()
    a = [] #initialize the empty list 'a', where we will store the polarity scores of the individual lines
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
    neg = stat.mean(neg_float_new)
    neu = stat.mean(neu_float_new)
    pos = stat.mean(pos_float_new)
    comp = stat.mean(comp_float_new)
    x = pos
    y = neu
    z = neg
    my_list = [x,y,z,comp]
    print (str(my_list))
    return my_list

def scanfolder():
    list_of_paths=[]
    for path, dirs, files in os.walk('/Users/BenWeissman/FinalProject-VaderML'):
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

def build_profiles(file_path):
    df = pd.read_csv(file_path)

    # Build the News Outlet profiles. Each profile contains
    # positive, negative, neutral, and compound values for
    # each individual record in the News Outlet profile
    AlJazeera = (df[df['NewsSource'].str.contains('Al')])
    BBC = (df[df['NewsSource'].str.contains('BBC')])
    Breitbart = (df[df['NewsSource'].str.contains('Breitbart')])
    CNN = (df[df['NewsSource'].str.contains('CNN')])
    FOX = (df[df['NewsSource'].str.contains('FOX')])
    HuffPo = (df[df['NewsSource'].str.contains('HuffPo')])
    NBCNews = (df[df['NewsSource'].str.contains('NBC')])
    NPR = (df[df['NewsSource'].str.contains('NPR')])
    NYT = (df[df['NewsSource'].str.contains('NYT')])
    WashPo = (df[df['NewsSource'].str.contains('WashPo')])

    list_of_values = df.values.tolist()
    print(list_of_values)

def build_centroids(file_path):
    df = pd.read_csv(file_path)

    # Build the News Outlet profiles. Each profile contains
    # positive, negative, neutral, and compound values for
    # each individual record in the News Outlet profile
    AlJazeera = (df[df['NewsSource'].str.contains('Al')])
    BBC = (df[df['NewsSource'].str.contains('BBC')])
    Breitbart = (df[df['NewsSource'].str.contains('Breitbart')])
    CNN = (df[df['NewsSource'].str.contains('CNN')])
    FOX = (df[df['NewsSource'].str.contains('FOX')])
    HuffPo = (df[df['NewsSource'].str.contains('HuffPo')])
    NBCNews = (df[df['NewsSource'].str.contains('NBC')])
    NPR = (df[df['NewsSource'].str.contains('NPR')])
    NYT = (df[df['NewsSource'].str.contains('NYT')])
    WashPo = (df[df['NewsSource'].str.contains('WashPo')])

    # Creating Coordinates
    AlJazeera = AlJazeera['Positive'].mean(),AlJazeera['Neutral'].mean(),AlJazeera['Negative'].mean()
    BBC = BBC['Positive'].mean(),BBC['Neutral'].mean(),BBC['Negative'].mean()
    Breitbart = Breitbart['Positive'].mean(),Breitbart['Neutral'].mean(),Breitbart['Negative'].mean()
    CNN = CNN['Positive'].mean(),CNN['Neutral'].mean(),CNN['Negative'].mean()
    FOX = FOX['Positive'].mean(),FOX['Neutral'].mean(),FOX['Negative'].mean()
    HuffPo = HuffPo['Positive'].mean(),HuffPo['Neutral'].mean(),HuffPo['Negative'].mean()
    NBCNews = NBCNews['Positive'].mean(),NBCNews['Neutral'].mean(),NBCNews['Negative'].mean()
    NPR = NPR['Positive'].mean(),NPR['Neutral'].mean(),NPR['Negative'].mean()
    NYT = NYT['Positive'].mean(),NYT['Neutral'].mean(),NYT['Negative'].mean()
    WashPo = WashPo['Positive'].mean(),WashPo['Neutral'].mean(),WashPo['Negative'].mean()

    articles = []
    articles.append(AlJazeera)
    articles.append(BBC)
    articles.append(Breitbart)
    articles.append(CNN)
    articles.append(FOX)
    articles.append(HuffPo)
    articles.append(NBCNews)
    articles.append(NPR)
    articles.append(NYT)
    articles.append(WashPo)

    new_frame = pd.DataFrame(articles)
    Title = ['Positive', 'Neutral','Negative']
    Index = 'AlJazeera','BBC','Breitbart','CNN','FOX','HuffPo','NBCNews','NPR','NYT','WashPo'
    new_frame.columns = Title
    new_frame.index = Index
    new_frame.to_csv('me.csv')
    return new_frame

def print_centroids(frame):
    frame = frame.sort_values(by=['Negative'], ascending=True)
    print(frame)

def dataframe_to_sourceframe(src):
    file_path = '/Users/BenWeissman/FinalProject-VaderML/Frame.csv'
    dataset = pd.read_csv(file_path)
    the_list = dataset.values.tolist()
    my_list = []
    for i in range(0,len(the_list)):
        if the_list[i][5] == src:
            my_list.append(the_list[i])
        else:
            pass
    return my_list


class SourceFrameToListOfLists(object):

    def __init__(self,sourceframe):
        self.sourceframe = sourceframe
        self.positive = []
        self.neutral = []
        self.negative = []
        self.compound = []
        self.news_source = []
        self.src = sourceframe[0][5]
        self.temp_list = []
        self.list_of_points = []

    def show_about(self):
        print("Source: " + self.src)
        print("n = " + str(len(self.sourceframe)))

    def populate_and_return_list_of_points(self):
        for i in range(0,len(self.sourceframe)):
            self.positive.append(self.sourceframe[i][1])
            self.neutral.append(self.sourceframe[i][2])
            self.negative.append(self.sourceframe[i][3])
            self.compound.append(self.sourceframe[i][4])
            self.news_source.append(self.sourceframe[i][5])
        my_list = [self.positive,self.neutral,self.negative,self.compound,self.news_source]
        self.list_of_points = []
        self.temp_list = []
        for i in range(0,len(my_list[0])):
            self.temp_list = [my_list[0][i],my_list[1][i],my_list[2][i],my_list[3][i],my_list[4][i]]
            self.list_of_points.append(self.temp_list)
        return self.list_of_points

    def populate_and_return_lists_of_values(self):
        for i in range(0,len(self.sourceframe)):
            self.positive.append(self.sourceframe[i][1])
            self.neutral.append(self.sourceframe[i][2])
            self.negative.append(self.sourceframe[i][3])
            self.compound.append(self.sourceframe[i][4])
            self.news_source.append(self.sourceframe[i][5])
        my_list = [self.positive,self.neutral,self.negative,self.compound,self.news_source]
        return my_list

def plot_whole_dataframe(*args):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(args[0][0], args[0][1], args[0][2], c='RED')
    ax.scatter(args[1][0], args[1][1], args[1][2], c='ORANGE')
    ax.scatter(args[2][0], args[2][1], args[2][2], c='YELLOW')
    ax.scatter(args[3][0], args[3][1], args[3][2], c='GREEN')
    ax.scatter(args[4][0], args[4][1], args[4][2], c='BLUE')
    ax.scatter(args[5][0], args[5][1], args[5][2], c='PURPLE')
    ax.scatter(args[6][0], args[6][1], args[6][2], c='OLIVE')
    ax.scatter(args[7][0], args[7][1], args[7][2], c='GRAY')
    ax.scatter(args[8][0], args[8][1], args[8][2], c='MAROON')
    ax.scatter(args[9][0], args[9][1], args[9][2], c='BLACK')


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def form_final_frame(*args):
    final_frame = []
    for i in range(0,len(args)):
        final_frame.append(SourceFrameToListOfLists(dataframe_to_sourceframe(args[i])).populate_and_return_list_of_points())


if __name__=='__main__':
    #scanfolder()
    #print_centroids(build_centroids('/Users/BenWeissman/FinalProject-VaderML/Frame.csv'))
    #build_profiles('/Users/BenWeissman/FinalProject-VaderML/Frame.csv')
    #plot_whole_dataframe(dataframe_to_sourceframe('Al-Jazeera'),dataframe_to_sourceframe('BBC'),dataframe_to_sourceframe('Breitbart'),dataframe_to_sourceframe('CNN'),dataframe_to_sourceframe('FOX'),dataframe_to_sourceframe('HuffPo'),dataframe_to_sourceframe('NBCNews'),dataframe_to_sourceframe('NPR'),dataframe_to_sourceframe('NYT'),dataframe_to_sourceframe('WashPo'))
    #check_prev(dataframe_to_sourceframe('/Users/BenWeissman/FinalProject-VaderML/Frame.csv'))
    #plot_whole_dataframe(dataframe_to_sourceframe('/Users/BenWeissman/FinalProject-VaderML/Frame.csv'))
    #dataframe_to_sourceframe('/Users/BenWeissman/FinalProject-VaderML/Frame.csv','Al-Jazeera')
    #url_to_sentiment('https://www.cnn.com/2018/05/01/politics/stormy-daniels-midterm-elections/index.html')

    a = SourceFrameToListOfLists(dataframe_to_sourceframe('Al-Jazeera')).populate_and_return_lists_of_values()
    b = SourceFrameToListOfLists(dataframe_to_sourceframe('BBC')).populate_and_return_lists_of_values()
    c = SourceFrameToListOfLists(dataframe_to_sourceframe('Breitbart')).populate_and_return_lists_of_values()
    d = SourceFrameToListOfLists(dataframe_to_sourceframe('CNN')).populate_and_return_lists_of_values()
    e = SourceFrameToListOfLists(dataframe_to_sourceframe('FOX')).populate_and_return_lists_of_values()
    f = SourceFrameToListOfLists(dataframe_to_sourceframe('HuffPo')).populate_and_return_lists_of_values()
    g = SourceFrameToListOfLists(dataframe_to_sourceframe('NBCNews')).populate_and_return_lists_of_values()
    h = SourceFrameToListOfLists(dataframe_to_sourceframe('NPR')).populate_and_return_lists_of_values()
    i = SourceFrameToListOfLists(dataframe_to_sourceframe('NYT')).populate_and_return_lists_of_values()
    j = SourceFrameToListOfLists(dataframe_to_sourceframe('WashPo')).populate_and_return_lists_of_values()

    plot_whole_dataframe(a,b,c,d,e,f,g,h,i,j)
