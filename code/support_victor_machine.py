from torchvision.io import read_image

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

from torch.utils.data import Dataset, DataLoader


def get_image(article_id, path=None):
    id_string = '0'+str(article_id)
    folder = id_string[0:3]
    path = '../resized_128/'+folder+'/'+id_string+'.jpg'
    try:
        return read_image(path)
    except:
        return 


def plot_images(article_ids, rows=1, columns=1, figsize=(20,10)):
    fig = plt.figure(figsize=figsize)

    for i in range(36):
        fig.add_subplot(rows, columns, i+1)
        try:
            img = get_image(articles_ids[i])
            imgplot = plt.imshow(img.permute(1,2,0))
        except:
            pass
        
