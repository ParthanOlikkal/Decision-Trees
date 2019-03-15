import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#Load Dataset
url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv"
my_data = pd.read_csv(url, delimiter = ",")
my_data

#Check size
my_data.size

#Preprocessing
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

#Categorical values such as Sex or BP are not handled by Sklearn Decision Trees.
#So we need to convert these features to numerical values.
#pandas.get_dummies() - Convert categorical variable into dummy/indicator variables

from sklearn import preprocessing 
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW','NORMAL','HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL','HIGH'])
X[:,3] = le_Chol.transform(X[:3])

X[0:5]

#fill target variable
y = my_data["Drug"]
y[0:5]

#Setting Decision Tree
from sklearn.model_selection import train_test_split
#train_test_split returns 4 parameters and needs 3 parameters.
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X,y,test_size = 0.3, random_state = 3)
#Here X and y are arrays required before the split, test_size is the ratio of testing dataset and random_state ensures that we receive the same splits
print('Train set : ', X_trainset, y_trainset)
print('Test set : ', X_testset, y_testset)

#Modeling
drugTree = DecisionTreeClassifier(criterion = "entropy", max_depth = 4)
drugTree # it shows the default parameters, entropy is used as it shows the information gain

#training
drugTree.fit(X_trainset, y_trainset)

#Prediction
predTree = drugTree.predict(X_testcase)
print (predTree [0:5])
print (y_testset [0:5])

#Evaluation
from sklearn import metrics
import matplotlib.pyplot as plt
print ("Decision Trees's Accuracy : ", metrics.accuracy_score(y_testset, predTree))

#Evaluation without sklearn


#Visualization
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mping
from sklearn import tree
%matplotlib inline

dot_data = StringIO()
filename = "drugTree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out = tree.export_graphviz(drugTree, feature_names = featureNames, out_file = dot_data, class_names = np.unique(y_trainset), filled = True, special_characters = True, rotate = False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mping.imread(filename)
plt.figure(figsize = (100, 200))
plt.imshow(img, interpolation = 'nearest')
