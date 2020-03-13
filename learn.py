# In[0]: Header and load data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #for split the data

in_df=pd.read_csv("final_dataset.csv")
in_df.loc[in_df['Label'] > 0, 'Label'] = 1
in_df.drop("Unnamed: 0", axis=1, inplace = True)
X = in_df.drop("Label",axis=1)
y = in_df["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import torch

if torch.cuda.is_available():
    device = torch.device('cuda')

#from_numpy takes a numpy element and returns torch.tensor
X = torch.from_numpy(X_train.to_numpy()).type(torch.FloatTensor)
y = torch.from_numpy(y_train.to_numpy()).type(torch.LongTensor)
X = X.to(device)
y = y.to(device)
print("X", X.device)
print("y", X.device)

from sklearn.metrics import make_scorer, roc_curve #for eer
from scipy.optimize import brentq #for eer
from scipy.interpolate import interp1d #for eer


# In[2]: Calculate eer rate
def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


# In[1]: Model

import torch.nn as nn
import torch.nn.functional as F#our class must extend nn.Module
class MyClassifier(nn.Module):
    def __init__(self):
        super(MyClassifier,self).__init__()
        #Our network consists of 3 layers. 1 input, 1 hidden and 1 output layer
        #This applies Linear transformation to input data. 
        self.fc1 = nn.Linear(34,100)
        
        #This applies linear transformation to produce output data
        self.fc2 = nn.Linear(100,2)
        
    #This must be implemented
    def forward(self,x):
        #Output of the first layer
        x = self.fc1(x)
        #Activation function is Relu. Feel free to experiment with this
        x = torch.tanh(x)
        #This produces output
        x = self.fc2(x)
        return x
        
    #This function takes an input and predicts the class, (0 or 1)        
    def predict(self,x):
        #Apply softmax to output. 
        pred = F.softmax(self.forward(x),dim=1)
        ans = []
        #Pick the class with maximum weight
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)


#Initialize the model        
model = MyClassifier().cuda()
#Define loss criterion
criterion = nn.CrossEntropyLoss()
#Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Number of epochs
epochs = 10000
#List to store losses
losses = []
for i in range(epochs):
    #Precit the output for Given input
    y_pred = model.forward(X)
    #Compute Cross entropy loss
    loss = criterion(y_pred,y)
    #Add loss to the list
    losses.append(loss.item())
    #Clear the previous gradients
    optimizer.zero_grad()
    #Compute gradients
    loss.backward()
    #Adjust weights
    optimizer.step()

from sklearn.metrics import accuracy_score

X = torch.from_numpy(X_test.to_numpy()).type(torch.FloatTensor)
y = torch.from_numpy(y_test.to_numpy()).type(torch.LongTensor)
X = X.to(device)
y = y.to(device)

torch.save(model.state_dict(), "nn_model.pkl")

print('--------------The Accuracy of the model----------------------------')
print('The accuracy of the Neural Network is', round(accuracy_score(model.predict(X), y_test)*100,2))
print('--------------The Accuracy of the model----------------------------')
print('The EER value of the Neural Network is', round(calculate_eer(model.predict(X), y_test)*100,2))


def predict(x):
 #Convert into numpy element to tensor
 x = torch.from_numpy(x).type(torch.FloatTensor)
 #Predict and return ans
 ans = model.predict(x)
 return ans.numpy()