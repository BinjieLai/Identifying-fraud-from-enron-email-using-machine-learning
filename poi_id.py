#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 3: Create new feature(s) 
def computeFraction( poi_messages, all_messages ): # Define the faction computation function
    if poi_messages == "NaN" or all_messages == "NaN":
        fraction = 0
    else:
        fraction = float(poi_messages) / float(all_messages)
    return fraction

def main():
    
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
    features_list = ['poi'] # You will need to use more features

### Load the dictionary containing the dataset
    data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
    #for key in data_dict:
        #print key
    #print data_dict["BAXTER JOHN C"]
    for key in data_dict["BAXTER JOHN C"]:
        if key != 'poi' and key != 'email_address':
            features_list.append(key)
    print features_list
    print len(features_list)
### Task 2: Remove outliers
    data_dict.pop("TOTAL", 0)
   
    for name in data_dict:
       data_point = data_dict[name]
    #print len(data_dict[name])
       from_poi_to_this_person = data_point["from_poi_to_this_person"]
       to_messages = data_point["to_messages"]
       fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )   
       data_point["fraction_from_poi"] = fraction_from_poi

       from_this_person_to_poi = data_point["from_this_person_to_poi"]
       from_messages = data_point["from_messages"]
       fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    #print fraction_to_poi
       data_point["fraction_to_poi"] = fraction_to_poi
    #print len(data_dict[name])    
    #print data_dict[name]
    features_list.append('fraction_from_poi')
    features_list.append('fraction_to_poi')
    
### Store to my_dataset for easy export below.
    my_dataset = data_dict

### Extract features and labels from dataset for local testing
#from feature_format import featureFormat, targetFeatureSplit
    data = featureFormat(my_dataset, features_list, sort_keys = True)
 #   print data

    labels, features = targetFeatureSplit(data)
#print len(data_dict)

#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

### Univariate feature selection
    #print len(features[0])   
    #print features[0]
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(f_classif, k=5)
    selector.fit(features, labels)
    features = selector.transform(features)
    #print features[0]

    
### Try K-Fold in cross validation
    from sklearn.cross_validation import KFold
    kf = KFold(len(features), 6)
    for train_indices, test_indices in kf:
        features_train = [features[ii] for ii in train_indices]
        features_test = [features[ii] for ii in test_indices]
        labels_train = [labels[ii] for ii in train_indices]
        labels_test = [labels[ii] for ii in test_indices]


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score

    from sklearn.naive_bayes import GaussianNB
    clf_gnb = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.
    clf_gnb.fit(features_train, labels_train)
   

    from sklearn import tree
    clf_dt = tree.DecisionTreeClassifier(min_samples_split=2)
    clf_dt = clf_dt.fit(features_train, labels_train)
    pred_dt = clf_dt.predict(features_test)
    relative_importance = clf_dt.feature_importances_
    print "relative_importance: ", relative_importance
    
   
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

    test_classifier(clf_gnb, my_dataset, features_list)
    test_classifier(clf_dt, my_dataset, features_list)
    

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

#dump_classifier_and_data(clf, my_dataset, features_list)


if __name__ == "__main__":
    main()