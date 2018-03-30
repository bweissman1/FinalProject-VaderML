import requests
import pickle
import matplotlib.pyplot as plt
import pandas as pd




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
