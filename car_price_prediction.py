# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# %% [markdown]
# # importing Data set

# %%
cars=pd.read_csv("F:\FCAI-CU(4-1)\machine\Assignment1\car_data.csv")

# %%
cars.head()

# %%
yy=cars.price.to_numpy()
xx=cars.enginesize.to_numpy()


# %%
print(xx)

# %% [markdown]
# # scatter plots

# %%
cars.corr()["price"].sort_values()

# %%
x=cars.enginesize
b=cars.carwidth
c=cars.horsepower
d=cars.curbweight
target=cars.price


plt.xlabel("engine size")
plt.ylabel("price")
plt.scatter(x,target)
plt.show()

# %%
plt.xlabel("carwidth")
plt.ylabel("price")
plt.scatter(b,target)
plt.show()

# %%
plt.xlabel("horsepower")
plt.ylabel("price")
plt.scatter(c,target)
plt.show()

# %%
plt.xlabel("carsweight")# In[ ]:
plt.ylabel("price")
plt.scatter(d,target)
plt.show()

# %%
cars=cars[['carwidth','horsepower','curbweight','enginesize','price']]

cars.head()

# %% [markdown]
# **Shuffle the dataset**

# %%
cars = cars.sample(frac = 1)

# %% [markdown]
# # linear regression 

# %%
class Linear_Regression():

   def __init__( self, learning_rate, no_of_iterations ) :
          
        self.learning_rate = learning_rate
          
        self.no_of_iterations = no_of_iterations
        

    # fit function to train the model

   def fit( self, X, Y ) :
        # no_of_training_examples, no_of_features
          
        self.m, self.n = X.shape
          
        # initiating the weight and bias
          
        self.w = np.zeros( self.n )
          
        self.b = 0
          
        self.X = X
          
        self.Y = Y


        # implementing Gradient Descent for Optimization
                  
        for i in range( self.no_of_iterations ) :
              
            self.update_weights()
              
        
      
    # function to update weights in gradient descent
      
   def update_weights( self ) :
             
        Y_prediction = self.predict( self.X )
          
        # calculate gradients  
      
        dw = - ( 2 * ( self.X.T ).dot( self.Y - Y_prediction )  ) / self.m
       
        db = - 2 * np.sum( self.Y - Y_prediction ) / self.m 
          
        # updating the weights
      
        self.w = self.w - self.learning_rate * dw
      
        self.b = self.b - self.learning_rate * db
          
      
    # Line function for prediction:
      
   def predict( self, X ) :
      
        return X.dot( self.w ) + self.b

   

# %%
def calculate_cost(X, y,w,b): 
        m = X.shape[0]
        cost = 0.0
        for i in range(m):                                
            f_wb_i = np.dot(X[i],w) + b           #(n,)(n,) = scalar (see np.dot)
            cost = cost + (f_wb_i - y[i])**2       #scalar
        cost = cost / (2 * m)                      #scalar    
        return cost

# %% [markdown]
# # normalizing Data

# %%
df = cars.drop('price', axis=1)
df_norm = (df-df.min())/(df.max()-df.min())
df_norm = pd.concat((df_norm, cars.price), 1)
df_norm.head()


# %% [markdown]
# # Splitting Data set into training and testing sets

# %%
#Defined X value and y value , and split the data train
X = df_norm.drop(columns="price")           
y = df_norm["price"]    # y = price

# split the data train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# %%
print("X Train : ", X_train.shape)
print("X Test  : ", X_test.shape)
print("Y Train : ", y_train.shape)
print("Y Test  : ", y_test.shape)

# %%
model = Linear_Regression(learning_rate=0.002,no_of_iterations=5000)

# %%
model.fit(X_train, y_train)

# %%
# printing the parameter values ( weights & bias)
print('weight = ', model.w)
print('bias = ', model.b)

# %%
test_data_prediction = model.predict(X_test)
print(calculate_cost(X_test,y_test,model.w,model.b))
#print(test_data_prediction)

# %%
print(r2_score(y_test,test_data_prediction)*100)

# %%
print(test_data_prediction)

# %%
g = model.calculate_cost(X_test,y_test)
plt.scatter(g,model.no_of_iterations)
plt.xlabel("cost")
plt.ylabel("number of iterations")
plt.title("Cost vs number of iteration")
plt.show()

# %%

plt.scatter(X_test['carwidth'], y_test, color = 'red')
plt.scatter(X_train['carwidth'], y_train, color = 'blue')
plt.xlabel(' carwidth ')
plt.ylabel('Price')
plt.title(' Price vs carwidth')
plt.legend(['test data','train data'],loc='lower right')
plt.show()
#horsepower
plt.scatter(X_test['horsepower'], y_test, color = 'red')
plt.scatter(X_train['horsepower'], y_train, color = 'blue')
plt.xlabel(' horsepower ')
plt.ylabel('Price')
plt.title(' Price vs horsepower')
plt.legend(['test dara','train data'],loc='lower right')
plt.show()
#curbweight
plt.scatter(X_test['curbweight'], y_test, color = 'red')
plt.scatter(X_train['curbweight'], y_train, color = 'blue')
plt.xlabel(' curbweight ')
plt.ylabel('Price')
plt.title(' Price vs curbweight')
plt.legend(['test dara','train data'],loc='lower right')
plt.show()
#enginesize
plt.scatter(X_test['enginesize'], y_test, color = 'red')
plt.scatter(X_train['enginesize'], y_train, color = 'blue')
plt.xlabel(' enginesize ')
plt.ylabel('Price')
plt.title(' Price vs enginesize')
plt.legend(['test dara','train data'],loc='lower right')
plt.show()


