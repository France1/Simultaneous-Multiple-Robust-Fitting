import numpy as np

class LinearModel:
    '''
    '''
    
    def __init__(self, model, params=None, w=None):
        self.model = model
        self.params = params
        self.w = w

    def basis_functions(self, x):
        
        if self.model=='line':
            # X_i = [1, x_i]
            X = np.hstack([np.ones_like(x.reshape(-1,1)), 
                           x.reshape(-1,1)])
        if self.model=='sine':
            # X_i = [1, sin(x_i), cos(x_i)]
            k = 2*np.pi/(self.w-1)
            X = np.hstack([np.ones_like(x.reshape(-1,1)), 
                           np.sin(k*x.reshape(-1,1)), 
                           np.cos(k*x.reshape(-1,1))])
        if self.model=='cubic':
            # X_i = [1, x_i, x_i^2, x_i^3]
            X = np.hstack([np.ones_like(x.reshape(-1,1)), 
                           x.reshape(-1,1), 
                           x.reshape(-1,1)**2,
                           x.reshape(-1,1)**3])

        return X
    
    # def transform_params(self):
        
    #     if self.model=='sine':
    #         for i,p in enumerate(self.params):
    #             self.params[i] = [p[0],p[1]*np.sin(p[2]),p[1]*np.cos(p[2])]

    def predict_one(self, X, a):
        return np.dot(X,a)


    def predict(self,x,scale=0):
        
        X = self.basis_functions(x)
        if len(self.params)==1:
            y_pred = np.dot(X,self.params[0])+np.random.normal(0, scale, len(x))
            y_pred = y_pred .tolist()
        else:
            y_pred = []
            for a in self.params:
                y = np.dot(X,a)+np.random.normal(0, scale, len(x))
                y_pred.append(y.tolist())
        
        return y_pred
    
    def residuals(self,x,y):
        
        x_predict, y_predict = self.predict(x)
        errors = np.zeros((len(x),len(self.params)))
        for i in range(len(self.params)):
            errors[:,i] = y-np.array(y_predict[i])
            
        return errors