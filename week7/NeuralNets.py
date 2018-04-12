
# coding: utf-8

# In[1]:


from __future__ import division
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sklearn


# In[2]:


#Need to check if i have added the node to the list again


#Node for the nerwork
class Node:
    def __init__(self, nClasses):
        self.nClasses = nClasses
        self.hx = 0
        self.gx = 0
        self.gama = 0
        #self.weights = np.random.randn(nClasses + 1)
        self.weights = []
            
    
    def createWeights(self):
        for i in xrange(self.nClasses + 1):
            self.weights.append(float("%.2f" % np.random.uniform(-1,1.1)))
            
    #set the error value for node        
    def setErr(self, gama):
        self.gama = gama
    
    def getErr(self):
        return self.gama
    
    def setHx(self, hx):
        self.hx = hx
        
    def getGx(self): 
        return self.gx
    
    def getWeight(self, index):
        return self.weights[index]
    
    def setWeight(self, index, weight):
        self.weights[index] = weight
    
    def calGx(self):
        self.gx = 1 / (1 + np.exp(-self.hx))
        
        
    


# In[3]:


class Layer:
    def __init__(self, nNodes,classes):
        self.nNodes = nNodes
        self.node_list = []
        
        for i in xrange(self.nNodes): 
            self.node_list.append(Node(len(classes)));
        
    def feedForward(self, classes):
        self.data = []
        count = 0
        for node in self.node_list:
            node.createWeights()
            #bias node
            hx = -1 * node.weights[0]
            for i in xrange(len(classes)):
                temp = classes[0] * node.weights[i+1]
                hx += temp
            node.setHx(hx)
            node.calGx()
            self.node_list[count] = node
            self.data.append(node.getGx())
            count += 1
            
    def getNodeList(self):
        return self.node_list
            
    def getnNode(self):
        return self.nNodes
    
    def setNode(self,index, node):
        self.node_list[index] = node
    
    def getNode(self, index):
        return self.node_list[index]
    
    def getData(self):
        #return the feedfoward list
        return self.data
        


# In[4]:


class NeuralNet:
    def __init__(self, nLayers = 2):
        self.nLayers = nLayers
        #list of layers for the whole training data
        self.Layers = []

                
    def fit(self, train_data, train_target):
        self.train_target = train_target
        self.train_data = train_data
        for i in xrange(self.nLayers - 1):
            temp = Layer(3, train_data[0])
            temp.feedForward(train_data[0])
            self.Layers.append(temp)
        target_classes = len(set(self.train_target))
        if target_classes == 2:
            output_layer = Layer(1, self.Layers[0].getData())
        else:
            output_layer = Layer(target_classes, self.Layers[0].getData())
        output_layer.feedForward(self.Layers[0].getData())
        self.Layers.append(output_layer)


        
    def train(self):
        learning_rate = 0.1
        train_list = []
        counter = 0
        target_classes = len(set(self.train_target))

        for row in self.train_data:
            self.Layers[0].feedForward(row)
            self.Layers[1].feedForward(self.Layers[0].getData())
            
            if target_classes == 2:
                fire = self.Layers[1].getData()[0]
                if (fire >= .5):
                    train_list.append(1)
                else:
                    train_list.append(0)
                    
            else:
                fire = np.argmax(self.Layers[-1].getData())
                #print fire
                train_list.append(fire)
            
            
            #calculate error for the output layer
            ol_index = 0
            _node_runner = 0
            for node in self.Layers[-1].getNodeList():
                #if there the target is binary:
                if (target_classes == 2):
                    error = node.getGx() * (1 - node.getGx()) * (node.getGx() - self.train_target[counter])
            
                #other case
                else:
                    if ol_index == self.train_target[counter]:
                        error = node.getGx() * (1 - node.getGx()) * (node.getGx() - 1)
                    else:
                        error = node.getGx() * (1 - node.getGx()) * (node.getGx() - 0)
                    ol_index += 1
                node.setErr(error)
                self.Layers[-1].setNode(_node_runner, node)
                _node_runner += 1

            index = 1
            _node_runner = 0
            for node in self.Layers[0].getNodeList():
                error = node.getGx() * (1 - node.getGx())
                _sum = 0
                for node_k in self.Layers[1].getNodeList():
                    _sum += (node_k.getWeight(index) * node_k.getErr())
                error *= _sum
                node.setErr(error)
                self.Layers[0].setNode(_node_runner, node)
                _node_runner += 1
                index += 1
            
            #setting weight for the bias
            _node_runner = 0
            for node in self.Layers[1].getNodeList():
                newWeight = node.getWeight(0) - learning_rate * node.getErr() * -1
                node.setWeight(0, newWeight)
                self.Layers[1].setNode(_node_runner, node)
                _node_runner += 1

            _node_runner = 0
            for node in self.Layers[0].getNodeList():
                newWeight = node.getWeight(0) - learning_rate * node.getErr() * -1
                node.setWeight(0, newWeight)   
                self.Layers[0].setNode(_node_runner, node)
                _node_runner += 1
            #setting for the others
           
            for i in xrange(self.Layers[0].getnNode()):
                _node_runner = 0
                for node in self.Layers[1].getNodeList():
                    newWeight = node.getWeight(i + 1) - learning_rate * node.getErr() * self.Layers[0].getNode(i).getGx()
                    node.setWeight(i + 1, newWeight)
                    self.Layers[1].setNode(_node_runner, node)
                    _node_runner += 1
                    
                  
            for i in xrange(len(row)):
                _node_runner = 0  
                for node in self.Layers[0].getNodeList():
                    newWeight = node.getWeight(i + 1) - learning_rate * node.getErr() * row[i]
                    node.setWeight(i + 1, newWeight)
                    self.Layers[0].setNode(_node_runner, node)
                    _node_runner += 1
            counter += 1
        match = 0
        for i in xrange(len(train_list)):
            if train_list[i] == self.train_target[i]:
                match += 1
                
        
        return ("%.2f" % (match / len(self.train_target) * 100));    
             
            


# In[5]:


iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)
X_norm = sklearn.preprocessing.normalize(X_train)


# In[6]:


netWork = NeuralNet()
netWork.fit(X_norm, y_train)
accuracy_list = []
for i in xrange(200):
    accuracy =  netWork.train()
    accuracy_list.append(accuracy)


# In[7]:


accuracy_list


# In[8]:


df = pd.read_csv("https://raw.githubusercontent.com/LamaHamadeh/Pima-Indians-Diabetes-DataSet-UCI/master/pima_indians_diabetes.txt",header=None,skipinitialspace=True)
df[[1,2,3,4,5]] = df[[1,2,3,4,5]].replace(0, np.NAN)
df.dropna(inplace=True)


# In[9]:


values = df.values
X = values[:,0:8]
y = values[:,8]


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_norm = sklearn.preprocessing.normalize(X_train)


# In[11]:


netWork = NeuralNet()
netWork.fit(X_norm, y_train)
accuracy_list = []
for i in xrange(200):
    accuracy =  netWork.train()
    accuracy_list.append(accuracy)
accuracy_list

