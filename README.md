# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the needed packages
2. Read the txt file using read_csv
3. Use numpy to find theta,x,y values
4. To visualize the data use plt.plot

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: 
RegisterNumber:  
*/
#import files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("ex1.txt",header=None)

plt.scatter(df[0],df[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

"""
Take in a np array X,y,theta and generate the cost function of using theta as parameter in a linear regression model
"""
def computeCost(X,y,theta):
    m=len(y) #length of the training data
    h=X.dot(theta) #hypothesis
    square_err=(h-y)**2
    
    return 1/(2*m)*np.sum(square_err) #returning J

df_n=df.values
m=df_n[:,0].size
X=np.append(np.ones((m,1)),df_n[:,0].reshape(m,1),axis=1)
y=df_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta) #call the function

"""
Take in np array X,y and theta and update theta by taking num_iters gradient steps with learning rate of alpha 
return theta and the list of the cost of theta during each iteration
"""
def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[]
    
    for i in range(num_iters):
        predictions = X.dot(theta)
        error = np.dot(X.transpose(),(predictions -y))
        descent = alpha*(1/m )*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

#Testing the implementation
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(df[0],df[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="purple")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

"""
Takes in numpy array of x and theta and return the predicted value of y based on theta
"""
def predict(x,theta):
    predictions = np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000 , we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000 , we predict a profit of $"+str(round(predict2,0)))
```

## Output:
![linear regression using gradient descent](sam.png)
![image](https://user-images.githubusercontent.com/94165108/195106558-12e0d941-3a3d-4746-b227-37210df27d13.png)
![image](https://user-images.githubusercontent.com/94165108/195106593-8a6d5b84-fd48-4bbb-902d-628f0839f0ec.png)
![image](https://user-images.githubusercontent.com/94165108/195106627-7c2c9558-691f-4a7b-a6e9-285a3ca48ac2.png)
![image](https://user-images.githubusercontent.com/94165108/195106657-ceb13add-09a7-437c-8d77-527993c7999e.png)
![image](https://user-images.githubusercontent.com/94165108/195106740-e3ae795c-9368-40dc-a60c-fc4026da360a.png)
![image](https://user-images.githubusercontent.com/94165108/195106790-62cce815-6ab0-404a-b9c6-96646be7f1bc.png)
![image](https://user-images.githubusercontent.com/94165108/195106824-ca53bf44-4e25-4edb-b17d-fae6e376cab7.png)
![image](https://user-images.githubusercontent.com/94165108/195106863-d346be37-e67c-4a4a-96b9-da45d9c4f39f.png)
![image](https://user-images.githubusercontent.com/94165108/195106896-44c37a86-56fb-4b82-944d-e04baa5d35f1.png)
![image](https://user-images.githubusercontent.com/94165108/195106921-5b6a528f-2a93-4824-81b8-942ed1cf47f2.png)
![image](https://user-images.githubusercontent.com/94165108/195107004-fe4f9ffe-7409-41d6-88e7-1a97184ee4d6.png)
![image](https://user-images.githubusercontent.com/94165108/195107028-86a14e71-1d1b-4f7e-8153-1ddb9582e1f9.png)
![image](https://user-images.githubusercontent.com/94165108/195107075-88a460c0-ef8a-4d2f-b801-ff927a4311b6.png)
![image](https://user-images.githubusercontent.com/94165108/195107114-f4f58fdb-1136-4f7d-bbc8-90868cbcba1f.png)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
