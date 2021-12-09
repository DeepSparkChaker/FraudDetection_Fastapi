# -*- coding: utf-8 -*-
"""Preparation class"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np 


class ColumnsSelector(BaseEstimator, TransformerMixin):
    def __init__(self, positions):
        self.positions = positions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #return np.array(X)[:, self.positions]
        return X.loc[:, self.positions] 
########################################################################
class CustomLogTransformer(BaseEstimator, TransformerMixin):
    # https://towardsdatascience.com/how-to-write-powerful-code-others-admire-with-custom-sklearn-transformers-34bc9087fdd
    def __init__(self):
        self._estimator = PowerTransformer()

    def fit(self, X, y=None):
        X_copy = np.copy(X) + 1
        self._estimator.fit(X_copy)

        return self

    def transform(self, X):
        X_copy = np.copy(X) + 1

        return self._estimator.transform(X_copy)

    def inverse_transform(self, X):
        X_reversed = self._estimator.inverse_transform(np.copy(X))

        return X_reversed - 1  


class CustomImputer(BaseEstimator, TransformerMixin) : 
    def __init__(self, variable, by) : 
            #self.something enables you to include the passed parameters
            #as object attributes and use it in other methods of the class
            self.variable = variable
            self.by = by

    def fit(self, X, y=None) : 
        self.map = X.groupby(self.by)[variable].mean()
        #self.map become an attribute that is, the map of values to
        #impute in function of index (corresponding table, like a dict)
        return self

def transform(self, X, y=None) : 
    X[variable] = X[variable].fillna(value = X[by].map(self.map))
    #Change the variable column. If the value is missing, value should 
    #be replaced by the mapping of column "by" according to the map you
    #created in fit method (self.map)
    return X

    # categorical missing value imputer
class Mapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables, mappings):

        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.mappings)

        return X  
    

class CountFrequencyEncoder(BaseEstimator, TransformerMixin):
    #temp = df['card1'].value_counts().to_dict()
    #df['card1_counts'] = df['card1'].map(temp)
    def __init__(
        self,
        encoding_method: str = "count",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        keep_variable=True,
                  ) -> None:

        self.encoding_method = encoding_method
        self.variables = variables
        self.keep_variable=keep_variable

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the counts or frequencies which will be used to replace the categories.
        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the
            variables to be transformed.
        y: pandas Series, default = None
            y is not needed in this encoder. You can pass y or None.
        """
        self.encoder_dict_ = {}

        # learn encoding maps
        for var in self.variables:
            if self.encoding_method == "count":
                self.encoder_dict_[var] = X[var].value_counts().to_dict()

            elif self.encoding_method == "frequency":
                n_obs = float(len(X))
                self.encoder_dict_[var] = (X[var].value_counts() / n_obs).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # replace categories by the learned parameters
        X = X.copy()
        for feature in self.encoder_dict_.keys():
            if self.keep_variable:
                X[feature+'_fq_enc'] = X[feature].map(self.encoder_dict_[feature])
            else:
                X[feature] = X[feature].map(self.encoder_dict_[feature])
        return X[self.variables].to_numpy()
 
class FeaturesEngineerGroup(BaseEstimator, TransformerMixin):
    def __init__(self,groupping_method ="mean",
                   variables=  "amount",
                   groupby_variables = "nameOrig"                         
                 ) :
        self.groupping_method = groupping_method
        self.variables=variables
        self.groupby_variables=groupby_variables
        
    def fit(self, X, y=None):
        """
        Learn the mean or median of  amount of each client which will be used to create new feature for each unqiue client in order to undersatant thier behavior .
        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
        The training dataset. Can be the entire dataframe, not just the
        variables to be transformed.
        y: pandas Series, default = None
        y is not needed in this encoder. You can pass y or None.
        """
        self.group_amount_dict_ = {}
        #df.groupby('card1')['TransactionAmt'].agg(['mean']).to_dict()
        #temp = df.groupby('card1')['TransactionAmt'].agg(['mean']).rename({'mean':'TransactionAmt_card1_mean'},axis=1)
        #df = pd.merge(df,temp,on='card1',how='left')
        #target_mean = df_train.groupby(['id1', 'id2'])['target'].mean().rename('avg')
        #df_test = df_test.join(target_mean, on=['id1', 'id2'])
        #lifeExp_per_continent = gapminder.groupby('continent').lifeExp.mean()
        # learn mean/medain 
        #for groupby in self.groupby_variables:
         #   for var in self.variables:
        if self.groupping_method == "mean":
            self.group_amount_dict_[self.variables] =X.fillna(np.nan).groupby([self.groupby_variables])[self.variables].agg(['mean']).to_dict()
        elif self.groupping_method == "median":
            self.group_amount_dict_[self.variables] =X.fillna(np.nan).groupby([self.groupby_variables])[self.variables].agg(['median']).to_dict()
        else:
            print('error , chose mean or median')
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        #for col in self.variables:
         #   for agg_type in self.groupping_method:
        new_col_name =  self.variables+'_Transaction_'+ self.groupping_method
        X[new_col_name] = X[self.groupby_variables].map(self.group_amount_dict_[ self.variables][self.groupping_method])
        return X[new_col_name].to_numpy().reshape(-1,1)    
    
 
class FeaturesEngineerGroup2(BaseEstimator, TransformerMixin):
    def __init__(self,groupping_method ="mean",
                   variables=  "amount",
                   groupby_variables = "nameOrig"                         
                 ) :
        self.groupping_method = groupping_method
        self.variables=variables
        self.groupby_variables=groupby_variables
        
    def fit(self, X, y=None):
        """
        Learn the mean or median of  amount of each client which will be used to create new feature for each unqiue client in order to undersatant thier behavior .
        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
        The training dataset. Can be the entire dataframe, not just the
        variables to be transformed.
        y: pandas Series, default = None
        y is not needed in this encoder. You can pass y or None.
        """
        X = X.copy()
        self.group_amount_dict_ = {}
        #df.groupby('card1')['TransactionAmt'].agg(['mean']).to_dict()
        #temp = df.groupby('card1')['TransactionAmt'].agg(['mean']).rename({'mean':'TransactionAmt_card1_mean'},axis=1)
        #df = pd.merge(df,temp,on='card1',how='left')
        #target_mean = df_train.groupby(['id1', 'id2'])['target'].mean().rename('avg')
        #df_test = df_test.join(target_mean, on=['id1', 'id2'])
        #lifeExp_per_continent = gapminder.groupby('continent').lifeExp.mean()
        # learn mean/medain 
        #for groupby in self.groupby_variables:
         #   for var in self.variables:

        print('we have {} unique clients'.format(X[self.groupby_variables].nunique()))
        new_col_name =  self.variables+'_Transaction_'+ self.groupping_method    
        X[new_col_name] = X.groupby([self.groupby_variables])[[self.variables]].transform(self.groupping_method)
        X = X.drop_duplicates(['nameOrig'])
    
        self.group_amount_dict_ = dict(zip(X[self.groupby_variables], X[new_col_name]))
        del X
        #print('we have {} unique mean amount : one for each client'.format(len(self.group_amount_dict_)))
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        #for col in self.variables:
         #   for agg_type in self.groupping_method:
        new_col_name =  self.variables+'_Transaction_'+ self.groupping_method
        X[new_col_name] = X[self.groupby_variables].map(self.group_amount_dict_)
        return X[new_col_name].to_numpy().reshape(-1,1)   
    

class FeaturesEngineerCumCount(BaseEstimator, TransformerMixin):
    def __init__(self,group_one ="step",
                   group_two=  "nameOrig"                       
                 ) :
        self.group_one =group_one
        self.group_two=group_two
        
    def fit(self, X, y=None):
        """
        """
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        new_col_name =  self.group_two+'_Transaction_count'
        X[new_col_name] = X.groupby([self.group_one, self.group_two])[[self.group_two]].transform('count')
        return X[new_col_name].to_numpy().reshape(-1,1)
# Complete Preprocess Pipe :
# complete pipe :
# select the float/cat columns
#cat_feautres = X.select_dtypes(include=['object','category']).columns
#num_features = X.select_dtypes(exclude=['object','category']).columns
#Define vcat pipeline
features_cum_count=['step','nameOrig']
features_groupby_amount=['amount','nameOrig']
features_frequency_orig_dest=['nameOrig','nameDest']
features_cum_count_pipe = Pipeline([
                     ('transformer_Encoder', FeaturesEngineerCumCount())
                    ])
features_groupby_pipe = Pipeline([
                     ('transformer_group_amount_mean', FeaturesEngineerGroup2()),
                     ('transformer_group_scaler', PowerTransformer())
                    ])
features_frequency_pipe = Pipeline([
                     ('Encoder', CountFrequencyEncoder(variables=['nameOrig','nameDest'],encoding_method ="frequency", keep_variable=False))
                    ])
type_pipe= Pipeline([
                     ('transformer_Encoder', ce.cat_boost.CatBoostEncoder())
                    ])
num_features0=[  'amount',  'oldbalanceOrig', 'newbalanceOrig' ,'oldbalanceDest', 'newbalanceDest']
#Define vnum pipeline
num_pipe = Pipeline([
                     ('scaler', PowerTransformer()),
                    ])
#Featureunion fitting training data
preprocessor = FeatureUnion(transformer_list=[('cum_count', features_cum_count_pipe),
                                              ('mean_amount', features_groupby_pipe),
                                              ('frequency_dest_orig', features_frequency_pipe),
                                              ('trans_type', type_pipe),
                                              ('num', num_pipe)])
data_preparing= ColumnTransformer([
    ('cum_count', features_cum_count_pipe, features_cum_count ),
    ('mean_amount', features_groupby_pipe, features_groupby_amount ),
    ('frequency_dest_orig', features_frequency_pipe, features_frequency_orig_dest ),
    ('trans_type', type_pipe, ['type'] ),
    ('num', num_pipe, num_features0)
], remainder='drop')