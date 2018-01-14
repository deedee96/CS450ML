
# coding: utf-8

# In[78]:


from sklearn import datasets
from __future__ import division
iris = datasets.load_iris()

# Show the data (the attributes of each instance)
print(iris.data)

# Show the target values (in numeric format) of each instance
print(iris.target)

# Show the actual target names that correspond to each number
print(iris.target_names)


























# In[89]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)


# In[52]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
model = classifier.fit(X_train, y_train)
targets_predicted = model.predict(X_test)


# In[88]:


def myAccurary(predicted, target):
    count = 0;
    for i in xrange(len(target)):
        if predicted[i] == target[i]:
            count += 1
    return count / len(target) * 100;


# In[57]:


class HardCodedClassifier():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self;
    
    def predict(self, X_test):
        return [0 for i in xrange(len(X_test))]
        


# In[84]:


classifier = HardCodedClassifier()
model = classifier.fit(X_train, y_train)
targets_predicted = model.predict(X_test)


# In[87]:


print "%.2f" % myAccurary(targets_predicted, y_test), "%"

