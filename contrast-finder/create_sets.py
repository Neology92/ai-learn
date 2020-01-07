import random as rnd
import numpy as np

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
    RGB = np.random.randint(0, 256, size=(times, 3))
    return RGB

# Generates contrast color based on RGB list
# returns list of integers
def generate_BW(RGB):
    BW = []
    for i in range(RGB.__len__()):
        R = RGB[i][0]
        G = RGB[i][1]
        B = RGB[i][2]
        font_color = get_contrast_color(R,G,B)
        BW.append(font_color)
    return np.array([BW]).T


# Gets X and Y, and slices according to ratio
# returns train and test sets
def slice_train_test(X,Y,ratio):
    if(X.__len__() != Y.__len__()):
        raise ValueError("Lists must have the same size")
        return -1

    train_len = int(ratio * X.__len__())
    # print("train_len: ", train_len)

    X_train = X[0: train_len]
    Y_train = Y[0: train_len]

    X_test = X[train_len:]
    Y_test = Y[train_len:]

    # print("train:")
    # print(X_train.__len__())
    # print(Y_train.__len__())
    # print("\ntest:")
    # print(X_test.__len__())
    # print(Y_test.__len__())

    return X_train, Y_train, X_test, Y_test
    
# write lists of data to files
def write_to_files(X_train, Y_train, X_test, Y_test, path):
    # Train files
    file = open(f'{path}/train_input.txt',"w")
    for i in range(X_train.__len__()):
        file.write(str(X_train[i])+'\n')
    file.close()

    file = open(f'{path}/train_output.txt',"w")
    for i in range(Y_train.__len__()):
        file.write(str(Y_train[i])+'\n')
    file.close()

    # Test files
    file = open(f'{path}/test_input.txt',"w")
    for i in range(X_test.__len__()):
        file.write(str(X_test[i])+'\n')
    file.close()

    file = open(f'{path}/test_output.txt',"w")
    for i in range(Y_test.__len__()):
        file.write(str(Y_test[i])+'\n')
    file.close()


# reads data
# returns data as lists
def read_from_files(path):
    X_train_buff = []
    Y_train_buff = []
    X_test_buff = []
    Y_test_buff = []
    
    # Train files
    file = open(f'{path}/train_input.txt',"r")
    for line in file:
        RGB = list(map(int, line[1:-2].split()))
        X_train_buff.append(RGB)
    file.close()
   
    file = open(f'{path}/train_output.txt',"r")
    for line in file:
        RGB = list(map(int, line[1:-2].split()))
        Y_train_buff.append(RGB)
    file.close()
    

    # Test files
    file = open(f'{path}/test_input.txt',"r")
    for line in file:
        RGB = list(map(int, line[1:-2].split()))
        X_test_buff.append(RGB)
    file.close()

    file = open(f'{path}/test_output.txt',"r")
    for line in file:
        RGB = list(map(int, line[1:-2].split()))
        Y_test_buff.append(RGB)
    file.close()


    X_train = np.array(X_train_buff)
    Y_train = np.array(Y_train_buff)
    X_test = np.array(X_test_buff)
    Y_test = np.array(Y_test_buff)

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":

        

    SET_LEN = 100         # 1+
    import sys
    if(sys.argv[1]):
        SET_LEN = int(sys.argv[1])
    
    SLICE_RATIO = 0.2    # 0 - 1

    RGB = generate_RGB(SET_LEN)
    BW = generate_BW(RGB)

    X_train, Y_train, X_test, Y_test = slice_train_test(RGB, BW, SLICE_RATIO)
    write_to_files(X_train, Y_train, X_test, Y_test, "./data")
