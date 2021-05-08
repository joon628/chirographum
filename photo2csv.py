from PIL import Image
import numpy as np
import csv


img = np.array(Image.open("Jackie_Handwritten_Data/a.jpg"))
# Now to convert numpy array to csv file.

def csvWriter(fil_name, nparray):
  example = nparray.tolist()
  with open(fil_name+'.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(example)

csvWriter("a.csv", img)
