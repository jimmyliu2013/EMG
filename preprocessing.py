import sklearn
import pandas as pd
import numpy as np
import sys
import os
import shutil
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import variance
from bokeh.layouts import column
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.classification import confusion_matrix, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from statsmodels.sandbox.regression.kernridgeregress_class import plt_closeall
from astropy.visualization.hist import hist
from sqlalchemy.sql.expression import false
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

ROOT_DIR = "C:/Users/Administrator/Desktop/elec811/project/"
PROCESSED_DIR = ROOT_DIR + "Processed/"
FIG_DIR = ROOT_DIR + "Figures/"

HEADER = ["BB-EMG", "TB-EMG", "BR-EMG", "AD-EMG", "LES-EMG", "TES-EMG", "Hand-switch", "Box-switch", "Motion-sensor1", "Motion-sensor2", "Motion-sensor3", "Motion-sensor4", "Motion-sensor5", "Motion-sensor6"]

DO_PLOT = False
SUBJECT_TO_PLOT = 3
TRIAL_TO_PLOT = 3
LOAD_TO_PLOT = 10

SAMPLING_FREQUENCY = 1000

FEATURE_WINDOW_SIZE = 200
OVERLAP = 0.5  # 50% overlapping sliding window


BUTTERWORTH_ORDER = 4
BANDPASS_LOW_CUTTING_FREQUENCY = 20
BANDPASS_HIGH_CUTTING_FREQUENCY = 490

MIN_LENGTH_TO_LABEL = 1000  # only label the consecutive box switch signal longer than 500ms, remove the shaking (usually at start and end)


# extract features from sliding window here
def calculate_features_for_each_column(column_data):
    column_data.reset_index(drop=True, inplace=True)
    rms = np.sqrt(np.mean(column_data ** 2))
#     mean = column_data.mean()
#     max = column_data.max()
#     min = column_data.min()
#     med = column_data.median()
#     skew = column_data.skew()
#     kurt = column_data.kurt()
#     std = column_data.std()
#     iqr = column_data.quantile(.75) - column_data.quantile(.25)
#     f, p = scipy.signal.periodogram(column_data, 1000)
# #     print(p)
# #     mean_fre = 1000 * scipy.sum(f * p) / (scipy.sum(p)*1000)
#     max_energy_freq = np.nan
#     if p.size != 0 :
#         max_energy_freq = np.asscalar(f[np.argmax(p)])
# 
#     mean_freq = np.nan
#     if np.sum(p) != 0:
#         mean_freq = np.average(f, weights=p)
#     
#     median_freq = np.median(p)
#         
#     waveform_length = calculate_waveform_length(column_data)
#     zero_crossing = calculate_zero_crossing(column_data)
#     
#     return [rms, mean, max, min, med, skew, kurt, std, iqr, max_energy_freq, mean_freq, median_freq, waveform_length, zero_crossing]
    return [rms]


def calculate_waveform_length(column_data):
    sum = 0
    for i in range(0, column_data.size - 1):
        sum += np.abs(column_data[i+1] - column_data[i])
    return sum


def calculate_zero_crossing(column_data):
    zero_crossing = 0
    for i in range(0, column_data.size - 1):
        if ((column_data[i+1]) > 0 and column_data[i] < 0):
            zero_crossing += 1
        if ((column_data[i+1]) < 0 and column_data[i] > 0):
            zero_crossing += 1
    return zero_crossing


def get_new_dir_for_filtered_data(dir):
    return PROCESSED_DIR + "filtered/" + dir


def get_new_dir_for_labeled_data(dir):
    return PROCESSED_DIR + "labeled/" + dir


def get_new_dir_for_features(dir):
    return PROCESSED_DIR + "features/" + dir


def clean():
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)


def init_dir():
    clean()
    os.mkdir(PROCESSED_DIR)
    os.mkdir(PROCESSED_DIR + "filtered/")
    os.mkdir(PROCESSED_DIR + "labeled/")
    os.mkdir(PROCESSED_DIR + "features/")
    for subject in range(1, 6):
        dir = get_dir_name(subject)
        os.mkdir(get_new_dir_for_filtered_data(dir))
        os.mkdir(get_new_dir_for_labeled_data(dir))
        os.mkdir(get_new_dir_for_features(dir))
    
#*****************************************#
#      1.filter all the raw data file     #
#*****************************************#


def get_dir_name(subject):
#     return ROOT_DIR + "/S0" + str(subject) + " Raw Data 30 Files/"
    return "S0" + str(subject) + " Raw Data 30 Files/"


def get_file_name(subject, trial, load):
#     dir = get_dir_name(subject)
    return "LiftCycle_" + load + "kg" + trial + ".txt"
    

def filter_data(subject):
    dir = get_dir_name(subject)
    files = os.listdir(ROOT_DIR + dir)
    
    for file in files:
        trial = get_trial_from_file_name(file)
        load = get_load_from_file_name(file)
        file_path = ROOT_DIR + dir + file
        data = pd.read_csv(file_path, sep="\t", header=None)
        data.columns = HEADER
        
        plot_data(data, "raw_S" + str(subject) + "_T" + str(trial) + "_L" + str(load), data.shape[1], subject, trial, load)
        
        for i in range(0, 6):  # first 6 columns of emg signal
            data.iloc[:, i] = butterworth_filter(data.iloc[:, i])
        plot_data(data, "filtered_S" + str(subject) + "_T" + str(trial) + "_L" + str(load), data.shape[1], subject, trial, load)
        data.to_csv(get_new_dir_for_filtered_data(dir) + file, sep="\t", index=False)
        

def butterworth_filter(data_column):
    nyq = 0.5 * SAMPLING_FREQUENCY
    normal_low_cutoff = BANDPASS_LOW_CUTTING_FREQUENCY / nyq
    normal_high_cutoff = BANDPASS_HIGH_CUTTING_FREQUENCY / nyq

    b, a = signal.butter(BUTTERWORTH_ORDER, [normal_low_cutoff, normal_high_cutoff], 'bandpass', analog=False)
    filtered = signal.filtfilt(b, a, data_column)
    return filtered

#*****************************************#
#           2.label all file              #
#*****************************************#


def get_load_from_file_name(file):
    second_part = file.split("_")[1]
    if second_part.startswith("0"):
        return 0
    elif second_part.startswith("5"):
        return 5
    elif second_part.startswith("10"):
        return 10
    elif second_part.startswith("15"):
        return 15
    elif second_part.startswith("20"):
        return 20
    elif second_part.startswith("2pt5"):
        return 2.5
    else:
        raise ValueError('label error')


def get_trial_from_file_name(file):
    second_part = file.split("kg")[1]
    return int(second_part[0])


def label_data(subject):
    dir = get_dir_name(subject)
    filtered_dir = get_new_dir_for_filtered_data(dir)
    files = os.listdir(filtered_dir)
    for file in files:
        file_path = filtered_dir + file
        data = pd.read_csv(file_path, sep="\t")
        
        load = get_load_from_file_name(file)
        trial = get_trial_from_file_name(file)

        data["Subject"] = [subject] * data.shape[0]
        data["Trial"] = [get_trial_from_file_name(file)] * data.shape[0]
        data["Load"] = [-1] * data.shape[0]

        # label by box switch only
        column = data.iloc[:, 7]  # data["Box-switch"]
#         plt.clf()
#         plt.plot(column, color="red")
        offset = 0.25  # signal offset from on the platform
        sigma_of_lift_in_box_switch = 0.3
        base_start = np.mean(data.iloc[0:1000, 7])
        base_end = np.mean(data.iloc[-1000:, 7])
        base = min(base_start, base_end)
        median = column[column < base - offset].median()
        
        data.loc[((data["Box-switch"] < median + sigma_of_lift_in_box_switch) & (data["Box-switch"] > median - sigma_of_lift_in_box_switch)), "Load"] = load
        
        # remove shaking of the signal
        start_and_end_index_list = []  # list of (start, end)
        start = -1
        end = -1
        for idx, l in data["Load"].iteritems():
            if l == -1:
                if start != -1:
                    end = idx
            else:
                if start == -1:
                    start = idx
            if start != -1 and end != -1:
                if end - start >= MIN_LENGTH_TO_LABEL:
                    start_and_end_index_list.append((start, end))
                start = -1
                end = -1
        
        for tp in start_and_end_index_list:
#             print(tp)
            data.loc[tp[0]: tp[1], "Load"] = load + 1
        data.loc[data["Load"] < load + 1, "Load"] = -1
        data.loc[data["Load"] == load + 1, "Load"] = load
        
        plot_data(data, "labeled_S" + str(subject) + "_T" + str(trial) + "_L" + str(load), data.shape[1], subject, trial, load)
        
        data = data.loc[data["Load"] != -1].reset_index(drop=True, inplace=False)
        
        plot_data(data, "removed_platform_S" + str(subject) + "_T" + str(trial) + "_L" + str(load), data.shape[1], subject, trial, load)
        data.to_csv(get_new_dir_for_labeled_data(dir) + file, sep="\t", index=False)
        
#         plt.plot(data.iloc[:,6], color="green")
#         plt.plot(data.loc[:,"Load"] + 7, color="blue")
# #         plt.hlines(base-0.25, 0, 15000, color="green")
#         plt.hlines(median, 0, 15000, color="black")
#         plt.yticks(np.arange(-2, 29, 1))
#         plt.xticks(np.arange(0, 15000, 1000))
#         plt.grid()
#         plt.title("label")
#         figure = plt.gcf()
#         figure.set_size_inches(15, 8)
#         plt.savefig(FIG_DIR + "label/" + str(subject) + "_" + str(get_load_from_file_name(file)) + "_" + str(get_trial_from_file_name(file)) +".png", dpi=100)

#*****************************************#
#           3.feature extraction          #
#*****************************************#


def window_and_extract_features(data, subject, file):
    dir = get_dir_name(subject)
    feature_list = []
    start = 0
    end = data.shape[0]
    trial = get_trial_from_file_name(file)
    load = get_load_from_file_name(file)
    while True:
        if start + FEATURE_WINDOW_SIZE < end:
            row_list = [subject, trial, load]
            for channel in [0, 1, 2, 3, 4, 5]:  # only emg signals
                column_data = data.iloc[start:start + FEATURE_WINDOW_SIZE, channel]
                features = calculate_features_for_each_column(column_data)
                row_list.extend(features)
            feature_list.append(row_list)
            start = (int)(start + 1) #FEATURE_WINDOW_SIZE * (1 - OVERLAP))
        elif (start + 100 < end):  # if not enough data points in this window, same method to calculate features
            row_list = [subject, trial, load]
            for channel in [0, 1, 2, 3, 4, 5]:
                column_data = data.iloc[start:end, channel]
                features = calculate_features_for_each_column(column_data)
                row_list.extend(features)
            feature_list.append(row_list)
            break
        else:
            print("column length is not enough")
            break
    if len(feature_list) > 0:
        features = pd.DataFrame(feature_list)
        plot_data(data, "data_for_feature_extraction_S" + str(subject) + "_T" + str(trial) + "_L" + str(load), data.shape[1], subject, trial, load)
        plot_data(features, "features_S" + str(subject) + "_T" + str(trial) + "_L" + str(load), features.shape[1], subject, trial, load)
        features.to_csv(get_new_dir_for_features(dir) + file, sep="\t", index=False)
    

def feature_extraction(subject):
    dir = get_dir_name(subject)
    labeled_dir = get_new_dir_for_labeled_data(dir)
    files = os.listdir(labeled_dir)
    for file in files:
        file_name = labeled_dir + file
        data = pd.read_csv(file_name, sep="\t")
        data.dropna(inplace=True)
        window_and_extract_features(data, subject, file)

#*****************************************#
#           4.catenate features           #
#*****************************************#


def catenate_feature_from_same_subject(subject):
    dir = get_dir_name(subject)
    feature_dir = get_new_dir_for_features(dir)
    df = pd.DataFrame()
    files = os.listdir(feature_dir)
    for file in files:
        file_path = feature_dir + file
        data = pd.read_csv(file_path, sep="\t")
        df = pd.concat([df, data], axis=0, sort=False, ignore_index=True)
    plot_data(df, "features_before_normalization_for_s" + str(subject), df.shape[1], subject, None, None)
    df.to_csv(feature_dir + "s" + str(subject) + "_all.features", sep="\t", index=False)
    
            
def catenate_feature_from_all_subject():
    df = pd.DataFrame()
    for subject in range(1, 6):
        dir = get_dir_name(subject)
        feature_dir = get_new_dir_for_features(dir)
        file = feature_dir + "s" + str(subject) + "_all_normalized.features"
        data = pd.read_csv(file, sep="\t")
#         print(data.head(5))
        df = pd.concat([df, data], axis=0, sort=False, ignore_index=True)
#     df.columns = ["Subject", "Trial", "Load", "RMS_BB-EMG", "RMS_TB-EMG", "RMS_BR-EMG", "RMS_AD-EMG", "RMS_LES-EMG", "RMS_TES-EMG"]
    plot_data(df, "final", df.shape[1], None, None, None)
    df.to_csv(PROCESSED_DIR + "final.features", sep="\t", index=False)


#*****************************************#
#           5.normalization               #
#*****************************************#
def normalize_data(subject):
    dir = get_dir_name(subject)
    feature_dir = get_new_dir_for_features(dir)
    file = feature_dir + "s" + str(subject) + "_all.features"
    data = pd.read_csv(file, sep="\t")
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    data.iloc[:, 3:] = min_max_scaler.fit_transform(data.iloc[:, 3:])
    plot_data(data, "normalized_features_for_S" + str(subject), data.shape[1], subject, None, None)
    data.to_csv(feature_dir + "s" + str(subject) + "_all_normalized.features", sep="\t", index=False)


def plot_data(data, title, columns, subject, trial, load):
    if DO_PLOT and ((subject == None) or (subject == SUBJECT_TO_PLOT)) and ((trial == None) or (trial == TRIAL_TO_PLOT)) and ((load == None) or (load == LOAD_TO_PLOT)):
        fig, ax = plt.subplots(columns, 1, dpi=60)
        plt.subplots_adjust(hspace=0.3)
        fig.suptitle(title, size=18)
        fig.set_size_inches(40, 20)
        
        for i in range(0, columns):
            ax[i].plot(data.iloc[:, i], color="C" + str(i % 10), label=data.columns[i])
            ax[i].set_ylabel(data.columns[i], rotation=0, labelpad=50)
            ax[i].grid()
            
        plt.show()
#         fig = plt.gcf()
#         fig.set_size_inches(15, 8)
#         fig.savefig("C:/Users/Administrator/Desktop/elec811/project/test/" + title + ".png")


def KNN(X_train, X_test, y_train, y_test):
    print("training data shape: ", X_train.shape)
    print("################# KNN #################")
    model = KNeighborsClassifier(n_neighbors=9)
    
    scores = sklearn.model_selection.cross_val_score(model, X_train, y_train, cv=KFold(n_splits=10, shuffle=True), scoring='accuracy')
    print("KNN cross-validation Accuracy: %0.2f" % scores.mean())
    
    model.fit(X_train, y_train)
    test_predict = model.predict(X_test)
    print("report for KNN: ")
    report = sklearn.metrics.classification_report(y_test, test_predict, digits=4)
    print(report)
    print("KNN overall accuracy: " + str(sklearn.metrics.accuracy_score(y_test, test_predict)))
    print(confusion_matrix(y_test, test_predict))




pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 1000)

print("########################## start ##########################")

"""
check the folder "Processed" under ROOT_DIR for intermediate files.

"""
init_dir()

"""
bandpass between BANDPASS_LOW_CUTTING_FREQUENCY and BANDPASS_HIGH_CUTTING_FREQUENCY.

"""
print("\n ******************** filtering ********************\n")
for subject in range(1, 6):
    filter_data(subject)

"""
by the shape of box-switch, as shown by the professor's pdf. then the "on platform" part and the unstable part near the contraction are removed.
only signals that last longer than MIN_LENGTH_TO_LABEL are retained.

"""
print("\n ******************** labeling ********************\n")
for subject in range(1, 6):
    label_data(subject)

"""
a sliding window of size FEATURE_WINDOW_SIZE moves along a EMG column, and extract features from each window.
modify the function calculate_features_for_each_column for feature extraction.

"""
print("\n ******************** extracting features ********************\n")
for subject in range(1, 6):
    feature_extraction(subject)

# catenate data from the same subject
for subject in range(1, 6):
    catenate_feature_from_same_subject(subject)

"""
simple min-max normalization, on all the data from the same subject. 
"""
print("\n ******************** normalizing ********************\n")
for subject in range(1, 6):
    normalize_data(subject)

# catenate all
catenate_feature_from_all_subject()
print("\n ******************** see " + PROCESSED_DIR + "final.features ********************\n")

labels = None
without_labels = None
    
data = pd.read_csv(PROCESSED_DIR + "final.features", sep="\t")
data.dropna(inplace=True)

lab_enc = sklearn.preprocessing.LabelEncoder()

without_labels = data.iloc[:, 3:]
labels = lab_enc.fit_transform(data.iloc[:, 2])

print(pd.unique(data.iloc[:, 2]))
X_train, X_test, y_train, y_test = train_test_split(without_labels, labels, test_size=0.2)
KNN(X_train, X_test, y_train, y_test)

print("########################## done ##########################")
