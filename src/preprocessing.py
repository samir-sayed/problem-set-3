'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd
import ast
from collections import defaultdict

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    
    model_pred_df = pd.read_csv('data/prediction_model_03.csv') #load csv
    genres_df = pd.read_csv('data/genres.csv') #load csv
    
    return model_pred_df, genres_df
    


def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
        
    '''
    
  
    genre_list = genres_df['genre'].unique().tolist() #create a list of all of the genres from the genre dataframe
    
    #initialize dictionaries to prepare for counting
    genre_true_counts = defaultdict(int)
    genre_tp_counts = defaultdict(int)
    genre_fp_counts = defaultdict(int)
    
    #check for initialization
    for genre in genre_list:
        genre_true_counts[genre] = 0
        genre_tp_counts[genre] = 0
        genre_fp_counts[genre] = 0

    #process
    for _, row in model_pred_df.iterrows():
        true_genres = ast.literal_eval(row['actual genres'])
        pred_genre = row['predicted']
        
        #had issues with sometimes not being able to handle as a list, this checks for it
        if not isinstance(true_genres, list):
            true_genres = [true_genres]

        
        for true_genre in true_genres:
            if true_genre in genre_true_counts:
                genre_true_counts[true_genre] += 1
        
        #update positiives and negatives
        if pred_genre in true_genres:
            genre_tp_counts[pred_genre] += 1
        else:
            genre_fp_counts[pred_genre] += 1
    
    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts
   
