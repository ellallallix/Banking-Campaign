#%% Pipeline / Youness


from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


preprocess = ColumnTransformer([
    ( ("num") , StandardScaler()   , ["age", "pdays", "previous", "campaign" , "nr.employed" , "euribor3m" , "cons.conf.idx" , "cons.price.idx" , "emp.var.rate"]),
    ( ("cat_non_ord") ,  OneHotEncoder(handle_unknown = 'ignore') , ["job", "marital", "education", "default", "housing",
            "loan", "contact", "month", "day_of_week", "poutcome"] ),
])




def pipeline(model):
    pipeline = Pipeline([("preprocess",preprocess),("model",model)])    
    return pipeline



#%%
