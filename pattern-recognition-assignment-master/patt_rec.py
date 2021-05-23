"""
DATASET:

-Each row is a number, each column is a pixel
-The dataset has 42001 rows and 785 columns
-The first row are the labels: label, pixel0, pixel1, ..., pixel783
-The first column indicate the right answer: the recognized digit
-From there, each row represent a number with 28x28 resolution (namely, 28x28 = 784 pixels)
-Since pixels starts from 0, the last pixel of a digit is named 'pixel783' and not 784
-All the 784 pixels are on the same row, but ideally, every 28 pixel should be another 'row' of the same digit

Hence, in the dataset is written as:
000 001 002 ... 027 028 029 ... 782 783

But this should be interpreted (to display the image of the digit) as:
000 001 002 003 ... 026 027
028 029 030 031 ... 054 055
056 057 058 059 ... 082 083
 |   |   |   |  ...  |   |
728 729 730 731 ... 754 755
756 757 758 759 ... 782 783

-Each pixel has a value from 0 to 255 (inclusive) indicating the lightness or darkness of that pixel, with higher numbers meaning darker

More information on: http://www.cs.uu.nl/docs/vakken/mpr/data/mnist-documentation.txt
Questions of the assignment: http://www.cs.uu.nl/docs/vakken/mpr/computer-labPython.php

Import the mirror feature with 100% margin: file named 'mirror_feature_100.0_.txt' on GitLab

Program made by Di Grandi Daniele and Matthijs Wolters
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC
import numpy as np
from mlxtend.evaluate import paired_ttest_5x2cv


def read_data(path):
    # read data from dataset stored locally in 'path'
    mnist_data_original = pd.read_csv(path)
    mnist_data = mnist_data_original.values

    return mnist_data_original, mnist_data


def get_data(mnist_data):
    # function to return a list with the solution (the recognized digit (from the data)) and store the pixels in a matrix
    return mnist_data[:, 0], mnist_data[:, 1:]


def plot_digit(digits, n):
    # n is the number of the row in the dataset that we want to plot, hence, could go from 0 to 41999
    img_size = 28
    plt.imshow(digits[n].reshape(img_size, img_size), cmap='gray_r')
    plt.show()


def count_digit_occurrence(mnist_data_original):
    # this function will count how many times each digit pop up in the dataset
    digit_occurrence = {'0': 0,
                        '1': 0,
                        '2': 0,
                        '3': 0,
                        '4': 0,
                        '5': 0,
                        '6': 0,
                        '7': 0,
                        '8': 0,
                        '9': 0}

    for i in range(10):
        digit_occurrence[str(i)] += (int(mnist_data_original.loc[mnist_data_original['label'] == i].groupby(['label']).count()['pixel0']))

    return digit_occurrence


def menu():

    print('\n0 -- Plot a digit from data')
    print('1 -- Run: Exploratory analysis and predict the majority class from probability distribution')
    print('2 -- Run: Ink feature')
    print('3 -- Run: Mirror feature (or tune the right percentage)')
    print('4 -- Run: Merged Ink + Mirror feature')
    print('5 -- Run: Point 5 on assignment')
    print('6 -- t test for point 5 model comparison (LR, SVM, NN)')
    print('e -- EXIT')

    choice = input('Choice: ')

    return choice


def get_distribution(mnist_data_original, data_rows):
    # compute the discrete distribution of each digit in the dataset
    digit_occurrence = count_digit_occurrence(mnist_data_original)

    probability_distribution = {'0': 0,
                                '1': 0,
                                '2': 0,
                                '3': 0,
                                '4': 0,
                                '5': 0,
                                '6': 0,
                                '7': 0,
                                '8': 0,
                                '9': 0}

    for i in range(10):
        probability_distribution[str(i)] += digit_occurrence[str(i)] / data_rows

    return probability_distribution


def get_class_and_probability(probability_distribution):
    # find the majority class and its probability from the distribution, that is, it's accuracy if always predicted
    max_probability = probability_distribution.get(max(probability_distribution, key=probability_distribution.get))
    class_predicted = int([k for k, v in probability_distribution.items() if v == max_probability][0])

    return max_probability, class_predicted


def show_output(probability_distribution, max_probability, class_predicted, useless_columns):
    # function to show the output of the 'predict the majority class' algorithm and the exploratory analysis of the dataset
    print('\nThe probability distribution for each class would be as follows:')

    for i in range(10):
        print(i, ':', round(probability_distribution[str(i)], 4) * 100, '%')

    print('\nMajority class always predicted:', class_predicted)
    print('Percentage of cases classified correctly:', round(max_probability, 4) * 100, '%')
    print('\nThe pixels with always value of zero (that is, always white) are:')
    for elem in useless_columns:
        print(elem)
    print('Then,', len(useless_columns), 'pixels could perhaps be eliminated from the analysis because not differential for classify the digit')


def find_useless_columns(mnist_data_original, mode=0):
    # if a pixel is always 0, it may be useless hence not differential when it comes to predict the class
    # if mode is 1 then recalculate the useless pixels, otherwise return the ones previously computed
    useless_columns = {}
    flag = 0

    if mode == 0:
        useless_columns = {'pixel0': 1, 'pixel1': 2, 'pixel2': 3, 'pixel3': 4, 'pixel4': 5, 'pixel5': 6, 'pixel6': 7, 'pixel7': 8, 'pixel8': 9, 'pixel9': 10, 'pixel10': 11, 'pixel11': 12, 'pixel16': 17, 'pixel17': 18, 'pixel18': 19, 'pixel19': 20, 'pixel20': 21, 'pixel21': 22, 'pixel22': 23, 'pixel23': 24, 'pixel24': 25, 'pixel25': 26, 'pixel26': 27, 'pixel27': 28, 'pixel28': 29, 'pixel29': 30, 'pixel30': 31, 'pixel31': 32, 'pixel52': 53, 'pixel53': 54, 'pixel54': 55, 'pixel55': 56, 'pixel56': 57, 'pixel57': 58, 'pixel82': 83, 'pixel83': 84, 'pixel84': 85, 'pixel85': 86, 'pixel111': 112, 'pixel112': 113, 'pixel139': 140, 'pixel140': 141, 'pixel141': 142, 'pixel168': 169, 'pixel196': 197, 'pixel392': 393, 'pixel420': 421, 'pixel421': 422, 'pixel448': 449, 'pixel476': 477, 'pixel532': 533, 'pixel560': 561, 'pixel644': 645, 'pixel645': 646, 'pixel671': 672, 'pixel672': 673, 'pixel673': 674, 'pixel699': 700, 'pixel700': 701, 'pixel701': 702, 'pixel727': 728, 'pixel728': 729, 'pixel729': 730, 'pixel730': 731, 'pixel731': 732, 'pixel754': 755, 'pixel755': 756, 'pixel756': 757, 'pixel757': 758, 'pixel758': 759, 'pixel759': 760, 'pixel760': 761, 'pixel780': 781, 'pixel781': 782, 'pixel782': 783, 'pixel783': 784}

    elif mode == 1:
        for i in range(784):
            for j in range(42000):
                print(i)
                if mnist_data_original['pixel' + str(i)][j] != 0:
                    flag = 1
                    break
            if flag == 0:
                useless_columns['pixel' + str(i)] = i + 1
            flag = 0

    return useless_columns


def ink_feature(digits, labels, mode=0):
    # function to compute the sum of the used ink for each digit (row) in the dataset and return this sum, with also the average and std dev of ink used per digit, toghether with an array of ordered data for the boxplot
    # if mode = 1 then activate the export function

    print('Summing the ink for each digit...')
    ink = np.array([sum(row) for row in digits])

    if mode == 1:
        export = pd.DataFrame(ink, columns=['Ink'])
        path = input('\nInsert the path to export the ink feature: ')
        export.to_csv(path, sep='\t', index=False)

    #compute the ink array for the boxplot:
    ink_for_boxplot = [[], [], [], [], [], [], [], [], [], []]

    for i in range(len(ink)):
        ink_for_boxplot[labels[i]].append(ink[i])

    #compute the mean and std dev
    ink_mean = [np.mean(ink[labels == i]) for i in range(10)]
    ink_std = [np.std(ink[labels == i]) for i in range(10)]

    return ink_mean, ink_std, ink, ink_for_boxplot


def merge_dataframes(ink, mirror_df):

    mirror_df['Ink'] = ink

    return mirror_df


def mirror_feature(mode, digits, margin=0.1):

    if mode == 0:

        path = input('\nPath to open the mirror feature file: ')
        mirror = pd.read_csv(path, sep='\t')

        return mirror

    elif mode == 1:

        mirror = []

        s = 0
        for digit in digits:
            mirror_array = [0, 0, 0, 0, 0, 0]
            s += 1
            for r in range(0, 14):
                for c in range(0, 14):
                    if digit[c+(r*28)] >= (1-margin) * digit[27-c+(r*28)] and digit[c+(r*28)] <= (1+margin) * digit[27-c+(r*28)]: #a to b
                        mirror_array[0]+=1
                    if digit[c+(r*28)] >= (1-margin) * digit[756+c-(r*28)] and digit[c+(r*28)] <= (1+margin) * digit[756+c-(r*28)]: #a to c
                        mirror_array[1] += 1
                    if digit[c+(r*28)] >= (1-margin) * digit[783-c-(r*28)] and digit[c+(r*28)] <= (1+margin) * digit[783-c-(r*28)]: #a to d
                        mirror_array[2] += 1
                    if digit[14+c+(r*28)] >= (1-margin) * digit[769-c-(r*28)] and digit[14+c+(r*28)] <= (1+margin) * digit[769-c-(r*28)]: #b to c
                        mirror_array[3] += 1
                    if digit[14+c+(r*28)] >= (1-margin) * digit[770+c-(r*28)] and digit[14+c+(r*28)] <= (1+margin) * digit[770+c-(r*28)]: #b to d
                        mirror_array[4] += 1
                    if digit[392+c+(r*28)] >= (1-margin) * digit[419-c+(r*28)] and digit[392+c+(r*28)] <= (1+margin) * digit[419-c+(r*28)]: #c to d
                        mirror_array[5] += 1
            norm_array = [x / 196 for x in mirror_array]
            mirror.append(norm_array)
            print('Margin:', margin, ', Remaining digits: ', 42000 - s)

        export_MF = pd.DataFrame(mirror, columns=['1', '2', '3', '4', '5', '6'])

        path = input('\nInsert the path to export the mirror feature obtained: ')
        export_MF.to_csv(path, sep='\t', index=False)

        return export_MF


def tune_percentage(digits, labels):

    acc_scores = []
    p_range = [x / 100 for x in range(1, 101)]

    for p in p_range:
        print('Evaluating:', p)
        mirror = mirror_feature(1, digits, margin=p)
        mirror = scale(mirror)
        model = LogisticRegression(random_state=0, multi_class='multinomial', penalty='none', solver='newton-cg').fit(mirror, labels)
        preds = model.predict(mirror)

        acc_scores.append((metrics.accuracy_score(labels, preds)))

    print('Accuracy for percentage:', acc_scores)


def output_model(model, preds, y_test, logistic=0):
    # function to print the output of the three models used

    if logistic == 1:
        print('\nIntercept: \n', model.intercept_)
        print('\nCoefficients: \n', model.coef_)

    # confusion matrix:
    confmtrx = np.array(confusion_matrix(y_test, preds))
    print('Complete confusion matrix:\n', confmtrx)
    export_confmtrx = pd.DataFrame(confmtrx, index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], columns=['pred 0', 'pred 1', 'pred 2', 'pred 3', 'pred 4', 'pred 5', 'pred 6', 'pred 7', 'pred 8', 'pred 9'])
    path = input('\nInsert the path to export the confusion matrix: ')
    export_confmtrx.to_csv(path, sep='\t')
    print('\nIndexed confusion matrix:\n')
    print(export_confmtrx)

    # accuracy and other reports:
    print('\nAccuracy Score:', round(metrics.accuracy_score(y_test, preds), 4) * 100, '%')
    class_report = classification_report(y_test, preds)  # insert: labels=np.unique(preds) in the class_report arguments for remove the warning, but it removes also the digits that were predicted 0 times
    print('\nDetailed report:')
    print(class_report)


def logistic_ink_feature(ink, labels):
    # logistic regression with only the ink feature
    ink = scale(ink).reshape(-1, 1)  # reshape is necessary for the logistic regression with a single feature

    # run the multinomial logistic regression:
    print('\nSolving the logistic regression...')
    model = LogisticRegression(random_state=0, multi_class='multinomial', penalty='none', solver='newton-cg').fit(ink, labels)

    preds = model.predict(ink)

    output_model(model, preds, labels, logistic=1)


def show_mean_stddev(ink_mean, ink_std):
    # function to print the mean and std dev of the ink cost for each digit
    print('\nDigit:\t\tMean:\t\tStd Dev:')
    for i in range(10):
        print(i, '\t\t', round(ink_mean[i], 2), '\t\t', round(ink_std[i], 2))


def boxplot_ink_feature(ink_for_boxplot):
    # plot a boxplot that describe the data used for the ink feature
    fig = plt.figure()
    ax = fig.add_subplot(111)  # ax = fig.add_axes([0, 0, 1, 1])
    ax.boxplot(ink_for_boxplot, patch_artist=True, notch='True', labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    plt.title("INK FEATURE BOX PLOT")
    plt.show()


def remove_useless_pixels(mnist_data_original, useless_columns):
    # function that removes from the data the pixels that are likely to be useless
    mnist_data_original = mnist_data_original.drop(columns=['label'])

    for i in useless_columns:
        mnist_data_original = mnist_data_original.drop(i, axis=1)
        print('Removed:', i)

    return mnist_data_original


def tune_lambda(X_train, y_train):
    # function that tunes the lambda hyperparameter of the LASSO (l1) penalty

    # total number of arrays for each iteration: 10
    # total number of features for each array: 784
    # total number of features in general: 784x10 = 7840

    k_scores = []
    k_number_of_remained_features = []
    k_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.814, 0.815, 0.816, 0.817, 0.818, 0.819, 0.82, 0.823, 0.826, 0.829, 0.83, 0.832, 0.835, 0.838, 0.84, 0.841, 0.844, 0.847, 0.85, 0.86, 0.87, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100, 200, 220, 260, 300, 320, 360, 400, 420, 460, 500, 520, 560, 600, 1000, 1500, 2000, 3000, 4000]

    for k in k_range:
        print('Testing:', k)
        model = LogisticRegression(random_state=0, multi_class='multinomial', penalty='l1', solver='saga', C=k)
        scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())

        model = LogisticRegression(random_state=0, multi_class='multinomial', penalty='l1', solver='saga', C=k).fit(X_train, y_train)

        y = list(model.coef_)
        zeros = 0

        for i in y:
            for j in range(len(i) - 1):
                if i[j] == 0:
                    zeros += 1

        k_number_of_remained_features.append(7840 - zeros)

    print('Accuracy for C:', k_scores)
    print('Remained features:', k_number_of_remained_features)

    plt.plot(k_range, k_scores)
    plt.xlabel('Value of C for Logistic Regression model with LASSO penalty')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()


def tune_params(X_train, y_train, parameters, model):

    grid = GridSearchCV(model, parameters, cv=10)
    grid_fit = grid.fit(X_train, y_train)
    print('Best parameters combination found:', grid_fit.best_params_)


def simple_logistic_regression(features, labels):

    features = scale(features)

    # run the multinomial logistic regression:
    print('\nSolving the logistic regression...')

    model = LogisticRegression(random_state=0, multi_class='multinomial', penalty='none', solver='newton-cg').fit(features, labels)
    preds = model.predict(features)
    output_model(model, preds, labels, logistic=1)


def optimal_parameters(mode):

    if mode == [1, 0, 0]:

        opt_C = 0.4

        return opt_C

    elif mode == [0, 1, 0]:

        opt_C = 50
        opt_degree = 3
        opt_gamma = 'scale'
        opt_kernel = 'poly'

        return opt_C, opt_degree, opt_gamma, opt_kernel

    elif mode == [0, 0, 1]:

        opt_alpha = 0.7
        opt_hidden_layer_sizes = (512, 3)
        opt_learning_rate = 'constant'
        opt_learning_rate_init = 0.001

        return opt_alpha, opt_hidden_layer_sizes, opt_learning_rate, opt_learning_rate_init


def solve_point_five(mode, mnist_data_original, labels, useless_columns):
    # mode is an array of type [1,0,0] or [1,1,0], ...
    # mode [1,0,0]: do only logistic regression with cross validation and lasso penalty
    # mode [0,1,0]: do only support vector machines
    # mode [0,0,1]: do only feed forward neural network
    # mode [0,1,1]: do both support vector machines and feed forward neural network
    # etc...

    remove_pixels = 0  # 0 for not remove the useless pixels, 1 for remove the useless pixels

    if remove_pixels == 0:
        mnist_data_original = mnist_data_original.drop(columns=['label'])
        features = mnist_data_original
    elif remove_pixels == 1:
        features = remove_useless_pixels(mnist_data_original, useless_columns)


    # scale the features:
    features = scale(features)

    # split data: (use random_state = 3 to produce always the same splitting of the data!) -> the same seed provided to random_state will produce the same splitting
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=37000, random_state=3)


    # choices on mode:
    if mode[0] == 1:

        tune_parameters = 0

        if tune_parameters == 0:
            # optimal hyperparameters:
            opt_C = optimal_parameters([1, 0, 0])

            # run the model:
            print('Solving the logistic regression...')

            model = LogisticRegression(random_state=0, multi_class='multinomial', penalty='l1', solver='saga', C=opt_C).fit(X_train, y_train)
            preds = model.predict(X_test)
            output_model(model, preds, y_test, logistic=1)

        elif tune_parameters == 1:
            tune_lambda(X_train, y_train)


    if mode[1] == 1:

        tune_parameters = 0

        if tune_parameters == 0:
            # optimal hyperparameters:
            opt_C, opt_degree, opt_gamma, opt_kernel = optimal_parameters([0, 1, 0])

            # run the model:
            print("Solving the support vector machine...")

            model = SVC(random_state=0, gamma=opt_gamma, kernel=opt_kernel, degree=opt_degree, C=opt_C).fit(X_train, y_train)
            preds = model.predict(X_test)
            output_model(model, preds, y_test)

        elif tune_parameters == 1:

            parameters = {
                "C": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.814, 0.815, 0.816, 0.817, 0.818, 0.819, 0.82, 0.823, 0.826, 0.829, 0.83, 0.832, 0.835, 0.838, 0.84, 0.841, 0.844, 0.847, 0.85, 0.86, 0.87, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100, 200, 220, 260, 300, 320, 360, 400, 420, 460, 500, 520, 560, 600, 1000, 1500, 2000, 3000, 4000],
                "kernel": ['linear', 'poly', 'rbf'],
                "degree": [3, 4, 5],
                "gamma": ['scale', 'auto']}

            model = SVC(random_state=0)
            tune_params(X_train, y_train, parameters, model)


    if mode[2] == 1:

        tune_parameters = 0

        if tune_parameters == 0:
            # optimal hyperparameters:
            opt_alpha, opt_hidden_layer_sizes, opt_learning_rate, opt_learning_rate_init = optimal_parameters([0, 0, 1])

            # run the model:
            print('Solving the neural network...')

            model = MLP(random_state=0, alpha=opt_alpha, hidden_layer_sizes=opt_hidden_layer_sizes, learning_rate=opt_learning_rate, learning_rate_init=opt_learning_rate_init).fit(X_train, y_train)
            preds = model.predict(X_test)
            output_model(model, preds, y_test)

        elif tune_parameters == 1:

            parameters = {
                "alpha": [1e-3, 0.1, 0.7, 0.9],
                "hidden_layer_sizes": [(300, 2), (397, 2), (500, 2), (800, 2), (512, 3)],
                "learning_rate": ['constant', "adaptive"],
                "learning_rate_init": [0.001, 0.1, 0.01, 0.0001],
                "max_iter": [300]}

            model = MLP(random_state=0)
            tune_params(X_train, y_train, parameters, model)


def solve_point_six(mnist_data_original, labels):

    mnist_data_original = mnist_data_original.drop(columns=['label'])

    opt_C_LR = optimal_parameters([1, 0, 0])
    opt_C_SVM, opt_degree, opt_gamma, opt_kernel = optimal_parameters([0, 1, 0])
    opt_alpha, opt_hidden_layer_sizes, opt_learning_rate, opt_learning_rate_init = optimal_parameters([0, 0, 1])

    LR_model = LogisticRegression(random_state=0, multi_class='multinomial', penalty='l1', solver='saga', C=opt_C_LR)
    SVM_model = SVC(random_state=0, gamma=opt_gamma, kernel=opt_kernel, degree=opt_degree, C=opt_C_SVM)
    NN_model = MLP(random_state=0, alpha=opt_alpha, hidden_layer_sizes=opt_hidden_layer_sizes, learning_rate=opt_learning_rate, learning_rate_init=opt_learning_rate_init)

    print('\nLR vs SVM...')
    t1, p1 = paired_ttest_5x2cv(estimator1=LR_model, estimator2=SVM_model, X=mnist_data_original, y=labels, random_seed=1, scoring='accuracy')
    print('\nt statistic (LR vs SVM): %.3f' % t1)
    print('p value (LR vs SVM): %.3f' % p1)

    print('\nLR vs NN...')
    t2, p2 = paired_ttest_5x2cv(estimator1=LR_model, estimator2=NN_model, X=mnist_data_original, y=labels, random_seed=1, scoring='accuracy')
    print('\nt statistic (LR vs NN): %.3f' % t2)
    print('p value (LR vs NN): %.3f' % p2)

    print('\nSVM vs NN...')
    t3, p3 = paired_ttest_5x2cv(estimator1=SVM_model, estimator2=NN_model, X=mnist_data_original, y=labels, random_seed=1, scoring='accuracy')
    print('\nt statistic (SVM vs NN): %.3f' % t3)
    print('p value (SVM vs NN): %.3f' % p3)


def get_mean(mirror, labels):
    # the mean is in format: [[0, ..., 0], ... [0, ..., 0]] where the inner list corresponds to a digit (they are 10) and the numbers within that corresponds to the feature (they are 6)
    # thus, a mean for each digit and for each feature value (for the mirror feature) is performed

    sum = [[0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0]]
    mean = [[0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0]]

    for l in range(42000):

        if labels[l] == 0:

            sum[0][0] += mirror['1'][l]
            sum[0][1] += mirror['2'][l]
            sum[0][2] += mirror['3'][l]
            sum[0][3] += mirror['4'][l]
            sum[0][4] += mirror['5'][l]
            sum[0][5] += mirror['6'][l]

        if labels[l] == 1:
            sum[1][0] += mirror['1'][l]
            sum[1][1] += mirror['2'][l]
            sum[1][2] += mirror['3'][l]
            sum[1][3] += mirror['4'][l]
            sum[1][4] += mirror['5'][l]
            sum[1][5] += mirror['6'][l]

        if labels[l] == 2:
            sum[2][0] += mirror['1'][l]
            sum[2][1] += mirror['2'][l]
            sum[2][2] += mirror['3'][l]
            sum[2][3] += mirror['4'][l]
            sum[2][4] += mirror['5'][l]
            sum[2][5] += mirror['6'][l]

        if labels[l] == 3:
            sum[3][0] += mirror['1'][l]
            sum[3][1] += mirror['2'][l]
            sum[3][2] += mirror['3'][l]
            sum[3][3] += mirror['4'][l]
            sum[3][4] += mirror['5'][l]
            sum[3][5] += mirror['6'][l]

        if labels[l] == 4:
            sum[4][0] += mirror['1'][l]
            sum[4][1] += mirror['2'][l]
            sum[4][2] += mirror['3'][l]
            sum[4][3] += mirror['4'][l]
            sum[4][4] += mirror['5'][l]
            sum[4][5] += mirror['6'][l]

        if labels[l] == 5:
            sum[5][0] += mirror['1'][l]
            sum[5][1] += mirror['2'][l]
            sum[5][2] += mirror['3'][l]
            sum[5][3] += mirror['4'][l]
            sum[5][4] += mirror['5'][l]
            sum[5][5] += mirror['6'][l]

        if labels[l] == 6:
            sum[6][0] += mirror['1'][l]
            sum[6][1] += mirror['2'][l]
            sum[6][2] += mirror['3'][l]
            sum[6][3] += mirror['4'][l]
            sum[6][4] += mirror['5'][l]
            sum[6][5] += mirror['6'][l]

        if labels[l] == 7:
            sum[7][0] += mirror['1'][l]
            sum[7][1] += mirror['2'][l]
            sum[7][2] += mirror['3'][l]
            sum[7][3] += mirror['4'][l]
            sum[7][4] += mirror['5'][l]
            sum[7][5] += mirror['6'][l]

        if labels[l] == 8:
            sum[8][0] += mirror['1'][l]
            sum[8][1] += mirror['2'][l]
            sum[8][2] += mirror['3'][l]
            sum[8][3] += mirror['4'][l]
            sum[8][4] += mirror['5'][l]
            sum[8][5] += mirror['6'][l]

        if labels[l] == 9:
            sum[9][0] += mirror['1'][l]
            sum[9][1] += mirror['2'][l]
            sum[9][2] += mirror['3'][l]
            sum[9][3] += mirror['4'][l]
            sum[9][4] += mirror['5'][l]
            sum[9][5] += mirror['6'][l]

    for p in range(10):

        mean[p] = [x / (labels == p).sum() for x in sum[p]]

    return mean


def make_table(array):

    print('Difference in mirrorness table:')

    print('\tf1\t\tf2\t\tf3\t\tf4\t\tf5\t\tf6\n')
    for i in range(10):
        print(i, '\t', end='')
        for j in range(6):
            print(round(array[i][j], 3), '\t', end='')
        print()


if __name__ == '__main__':

    # get the dataset, the labels and the digits
    path = input('\nInsert open path: ')

    mnist_data_original, mnist_data = read_data(path)
    labels, digits = get_data(mnist_data)
    data_rows = 42000
    mirror_percentage = 1 # it means: 100% of margin percentage in the mirror feature of point 3 and 4

    # start with the actual program
    choice = menu()

    while choice != 'e':

        if choice == '0':

            n = int(input('\nWhich position in the dataset you want to see the digit?: '))
            plot_digit(digits, n)
            print('The digit is:', labels[n])

        elif choice == '1':

            probability_distribution = get_distribution(mnist_data_original, data_rows)
            max_probability, class_predicted = get_class_and_probability(probability_distribution)
            useless_columns = find_useless_columns(mnist_data_original)  #this function will find the possible useless pixels: the ones with always value of 0
            show_output(probability_distribution, max_probability, class_predicted, useless_columns)


        elif choice == '2':

            ink_mean, ink_std, ink, ink_for_boxplot = ink_feature(digits, labels)
            show_mean_stddev(ink_mean, ink_std)
            logistic_ink_feature(ink, labels)
            boxplot_ink_feature(ink_for_boxplot)


        elif choice == '3':

            tune_parameter = 0  # set to 1 if you want to tune the percentage

            if tune_parameter == 0:
                mirror = mirror_feature(0, digits, margin=mirror_percentage)
                mean = get_mean(mirror, labels)

                maximum = 0
                minimum = 10000000
                for i in mean:
                    for j in range(6):
                        if i[j] > maximum:
                            maximum = i[j]
                        if i[j] < minimum:
                            minimum = i[j]

                mean_scaled = [[0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0]]
                for p in range(10):
                    mean_scaled[p] = [((x - minimum) / (maximum - minimum)) for x in mean[p]]

                make_table(mean_scaled)
                simple_logistic_regression(mirror, labels)

            elif tune_parameter == 1:
                tune_percentage(digits, labels)


        elif choice == '4':

            ink_mean, ink_std, ink, ink_for_boxplot = ink_feature(digits, labels)
            mirror = mirror_feature(0, digits, margin=mirror_percentage)
            mirror_and_ink = merge_dataframes(ink, mirror)
            simple_logistic_regression(mirror_and_ink, labels)


        elif choice == '5':

            useless_columns = find_useless_columns(mnist_data_original)
            solve_point_five([1, 1, 1], mnist_data_original, labels, useless_columns)


        elif choice == '6':

            solve_point_six(mnist_data_original, labels)


        else:
            print('\nInvalid choice')

        choice = menu()
