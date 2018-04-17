"""

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import requests
import pickle
import pandas as pd
import newspaper
import requests
from newspaper import Article

url = str(input('Enter your url here for awesomeness'))
file_name = str(input('Enter the name of the file you want to load text into'))


def get_book(url, file_string_name):
    ''' Loads book into a file and saves it as file_string_name'''
    article = Article(url)
    article.download()
    article.parse()
    book_doc = article.text
    f=open(file_string_name,'wb')
    pickle.dump(book_doc,f)
    f.close()

def open_book():
    ''' Opens and reads the book file and allows reader to print the book'''
    global file_name
    input_file=open(file_name,'rb')
    reloaded_copy_of_texts=pickle.load(input_file)
    print (reloaded_copy_of_texts)
    return reloaded_copy_of_texts
    
if __name__=='__main__':
    get_book(url,file_name)
    open_book()
