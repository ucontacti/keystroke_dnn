import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #for split the data

in_df=pd.read_csv("final_dataset.csv")
in_df.loc[in_df['Label'] > 0, 'Label'] = 1
in_df.drop("Unnamed: 0", axis=1, inplace = True)
X = in_df.drop("Label",axis=1)
y = in_df["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.metrics import accuracy_score

import torch 

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


if torch.cuda.is_available():
    device = torch.device('cuda')

model = MyClassifier().cuda()
model.load_state_dict(torch.load("nn_model.pkl"))
model.eval()


X = torch.from_numpy(X_test.to_numpy()).type(torch.FloatTensor)
y = torch.from_numpy(y_test.to_numpy()).type(torch.LongTensor)
X = X.to(device)
y = y.to(device)

print('--------------The Accuracy of the model----------------------------')
print('The accuracy of the Neural Network is', round(accuracy_score(model.predict(X), y_test)*100,2))
