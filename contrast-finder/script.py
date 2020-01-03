from create_sets import read_from_files

X_train, Y_train, X_test, Y_test = read_from_files("./data")

print(X_train.__len__(), Y_train.__len__(), X_test.__len__(), Y_test.__len__())