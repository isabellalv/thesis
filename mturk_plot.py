#PLOTTING DATA FROM MTURK CSV

import pandas as pd
import matplotlib.pyplot as plt

import ipdb 
ipdb.set_trace()

mturk_data = pd.read_csv('mturk_results_b0_cleaned.csv')

#mturk_data.Image_1



#plt.plot(mturk_data.Image_1, mturk_data.Similarity_Rating)
plt.plot(mturk_data.Image_2, mturk_data.Similarity_Rating)


plt.show()
