import numpy as np 
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def load_hog_data(hog_txt):
    """ load hog features.
    Args:
        hog_txt: a txt file used to save hog features.
            one line data is formated as "img_path \t cls_num \t hog_feature_vector"
    Return:
        img_names: a list of image names.
        labels: numpy array labels (1-dim).
        hog_feature: numpy array hog features.
            formated as [[hog1], [hog2], ...]
    """
    img_names = []
    labels = []
    hog_features = []
    with open(hog_txt, "r") as f:
        data = f.readlines()
        for row_data in data:
            row_data = row_data.rstrip()
            img_path, label, hog_str = row_data.split("\t")
            img_name = img_path.split("/")[-1]
            hog_feature = hog_str.split(" ")
            hog_feature = [float(hog) for hog in hog_feature]
            #print "hog feature length = ", len(hog_feature)
            img_names.append(img_name)
            labels.append(int(label))
            hog_features.append(hog_feature)
    return img_names, np.array(labels), np.array(hog_features)



def svm_train(hog_features, labels, save_path="./svm_model.pkl"):
    """ SVM train
    Args:
        hog_feature: numpy array hog features.
            formated as [[hog1], [hog2], ...]
        labels: numpy array labels (1-dim).
        save_path: model save path.
    Return:
        none.
    """
    clf = SVC(C=10, tol=1e-3, probability = True)
    clf.fit(hog_features, labels)
    joblib.dump(clf, save_path)
    print "finished."

def svm_test(svm_model, hog_feature, labels):
    """SVM test
    Args:
        hog_feature: numpy array hog features.
            formated as [[hog1], [hog2], ...]
        labels: numpy array labels (1-dim).
    Return:
        accuracy: test accuracy.
    """
    clf = joblib.load(svm_model)
    accuracy = clf.score(hog_feature, labels)
    return accuracy

def plot_roc_auc(clf, x_test):
    y_test = clf.predict(x_test)
    y_score = clf.decision_function(x_test)
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1], [1,0])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate')  
    plt.title('Receiver operating characteristic example')  

    






if __name__ =="__main__":
    hog_train_txt = "/home/meringue/Documents/traffic_sign_detection/data/proposals_train_hog.txt"
    hog_test_txt = "/home/meringue/Documents/traffic_sign_detection/data/proposals_test_hog.txt"
    model_path = "./svm_model.pkl"

    train = False

    if train is True:
        img_names, labels, hog_train_features = load_hog_data(hog_train_txt)
        svm_train(hog_train_features, labels, model_path)
    else:
        img_names, labels, hog_test_features = load_hog_data(hog_test_txt)
        test_accuracy = svm_test(model_path, hog_test_features, labels)
        print "test accuracy = ", test_accuracy

        #clf = joblib.load(model_path)
        #plot_roc_auc(clf, hog_test_features)
