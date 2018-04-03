"""
@author: John-Edwin Gadasu
"""

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import requests
import pickle
import pandas as pd


#Books to be analyzed
mythology_book_url='http://www.gutenberg.org/cache/epub/3327/pg3327.txt'
atheism_book_url='http://www.gutenberg.org/cache/epub/17607/pg17607.txt'
christianity_book_url='http://www.gutenberg.org/cache/epub/8247/pg8247.txt'
islam_book_url='http://www.gutenberg.org/cache/epub/2800/pg2800.txt'


def get_book(url, file_string_name):
    ''' Loads book into a file and saves it as file_string_name'''
    book_doc=requests.get(url).text
    f=open(file_string_name,'wb')
    pickle.dump(book_doc,f)
    f.close()

def open_book(file_string_name):
    ''' Opens and reads the book file and allows reader to print the book'''
    input_file=open(file_string_name,'rb')
    reloaded_copy_of_texts=pickle.load(input_file)
    return reloaded_copy_of_texts


def sentiment_analyzer(file_string_name):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    analyzer=SentimentIntensityAnalyzer()
    book_analysis=analyzer.polarity_scores(open_book(file_string_name))
    print (file_string_name)
    print (book_analysis)
    return book_analysis


def plot_3d_graph():
    '''Plots a bar graph of the degree of the
        various sentiments in subplots after a
        SentimentIntensityAnalyzer class is used'''

    get_book(atheism_book_url,'atheism.txt')
    get_book(christianity_book_url,'chrstn.txt')
    get_book(islam_book_url,'islam.txt')
    get_book(mythology_book_url,'myth.txt')

    atheism_analysis=sentiment_analyzer('atheism.txt')
    chrstn_analysis=sentiment_analyzer('chrstn.txt')
    islam_analysis=sentiment_analyzer('islam.txt')
    myth_analysis=sentiment_analyzer('myth.txt')

    list_of_analysis=[chrstn_analysis,atheism_analysis,islam_analysis, myth_analysis]

    b = dict()
    for key in list_of_analysis[0].keys():
        b[key] = []
        for dic in list_of_analysis:
            b[key].append(dic[key])

    b = pd.DataFrame(b)
    fig = plt.figure()
    ax = Axes3D(fig)
    x = ((list(b.iloc[:,1])))  #Negative
    y = ( (list(b.iloc[:,3]))) #Positive
    z=  ((list(b.iloc[:,0])))  #Compound
    ax.scatter(xs=x, ys=y, zs=z )
    ax.set_xlabel('Negative', rotation = 150)
    ax.set_ylabel('Positive')
    ax.set_zlabel('Compound', rotation = 60)

    plt.show()


if __name__=='__main__':
    plot_3d_graph()

plt.show()
