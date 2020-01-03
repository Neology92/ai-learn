import random as rnd

# Gets R,G,B and calculates brightness
# Checks which color is more contrast
# returns black or white (0 or 255)
def get_contrast_color(R,G,B):
    brightness = (299*R + 587*G + 114*B) / 1000
    if(brightness <= 127):
        return 255
    else:
        return 0

# Generates RGB x times
# returns list of dictionaries
def generate_RGB(times):
    RGB = []
    for i in range(times):
        R = rnd.randint(0,255)
        G = rnd.randint(0,255)
        B = rnd.randint(0,255)
        RGB.append({"R":R, "G":G, "B":B})
    return RGB

# Generates contrast color based on RGB list
# returns list of integers
def generate_BW(RGB):
    BW = []
    for i in range(RGB.__len__()):
        R = RGB[i]['R']
        G = RGB[i]['G']
        B = RGB[i]['B']
        font_color = get_contrast_color(R,G,B)
        BW.append(font_color)
    return BW


# Gets X and Y, and slices according to ratio
# returns train and test sets
def slice_train_test(X,Y,ratio):
    if(X.__len__() != Y.__len__()):
        raise ValueError("Lists must have the same size")
        return -1

    train_len = int(ratio * X.__len__())
    print("train_len: ", train_len)

    X_train = X[0: train_len]
    Y_train = Y[0: train_len]

    X_test = X[train_len:]
    Y_test = Y[train_len:]

    return X_train, Y_train, X_test, Y_test
    




if __name__ == "__main__":

    SET_LEN = 10        # 1+
    SLICE_RATIO = 0.3  # 0 - 1

    RGB = generate_RGB(SET_LEN)
    BW = generate_BW(RGB)

    print(RGB.__len__())
    print(BW.__len__())

    X_train, Y_train, X_test, Y_test = slice_train_test(RGB, BW, SLICE_RATIO)

    print ("X_train: ", X_train.__len__())
    print ("Y_train: ", Y_train.__len__())
    print ("X_test: ", X_test.__len__())
    print ("Y_test: ", Y_test.__len__())