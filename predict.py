import cv2
from sklearn.externals import joblib

def predict_digit(filename):
  # read trained data
    clf = joblib.load("digits.pkl")
    # read handwrite
    my_img = cv2.imread(filename)
    # convert image to acceable format
    my_img = cv2.cvtColor(my_img, cv2.COLOR_BGR2GRAY)
    my_img = cv2.resize(my_img, (8, 8))
    my_img = 15 - my_img // 16 # change black and white
    # 2d -> 1d
    my_img = my_img.reshape((-1, 64))
    # predict 
    res = clf.predict(my_img)
    return res[0]

# save image file and run
n = predict_digit("my2.png")
print("my2.png = " + str(n))
n = predict_digit("my4.png")
print("my4.png = " + str(n))
