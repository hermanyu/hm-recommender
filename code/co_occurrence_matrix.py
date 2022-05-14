#!/usr/bin/python

import sys

print("start date: ", sys.argv[1])
print("end date: ", sys.argv[2])

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

save_path = '../data/co_occurrence_matrix-'+str(sys.argv[1])+'_to_'+str(sys.argv[2])+'.csv'

print("loading transactions... ")

transactions = pd.read_csv('../data/transactions-by-product.csv', parse_dates = ['t_dat'])

selected = transactions[ ( transactions['t_dat'] > pd.to_datetime(sys.argv[1]) ) & ( transactions['t_dat'] < pd.to_datetime(sys.argv[2]) ) ]

print("consolidating transactions... ")

crosstab = pd.crosstab(index = selected['customer_id'], columns=selected['product_code'])

crosstab_sparse = csr_matrix(crosstab)

print("computing co-occurrence matrix... ")
co_occurrence = crosstab_sparse.transpose().dot(crosstab_sparse)

co_occurrence = co_occurrence.todense()

print("saving results to csv... ")

pd.DataFrame(co_occurrence, index=crosstab.columns, columns = crosstab.columns).to_csv(save_path)

print(" ")
print("finished!")