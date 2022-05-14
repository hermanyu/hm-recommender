from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(prod_code):
    prod_string = '0'+str(prod_code)
    path = '../images/'+prod_string[0:3]
    dir_list = os.listdir(path)
    for file_name in dir_list:
        if prod_string in file_name[0:7]:
            plt.imshow(mpimg.imread(path+'/'+file_name))
            break
            



with open('../models/pretrained-sbert.pkl', 'rb') as model_in:
    model = joblib.load(model_in)

product_embeddings = pd.read_csv('../data/full-desc-embeddings.csv', index_col = 'product_code')

user_input = input('Hello, what are you looking for? ')
user_embedding = model.encode([user_input])

similarities = []

for product in product_embeddings.index:
    similarities.append([product]+ list(cosine_similarity(user_embedding, np.array(product_embeddings.loc[product]).reshape(1,384))[0]))

df = pd.DataFrame(similarities, columns=['product_code', 'similarity'])
df.sort_values(by='similarity', ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)

print(df.head())

for i in range(10):
    show_image(df.iloc[i,0])
    