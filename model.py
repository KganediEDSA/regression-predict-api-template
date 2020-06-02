"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    # Importing the dataset
    dataset = pd.read_csv('Train.csv')
    dataset_test = pd.read_csv('Test.csv')
    
    # Selecting only the relevent columns that will be used for the model and replacing white space with _ on columns names
    new_df = dataset[['Order No','User Id','Platform Type','Personal or Business','Placement - Day of Month','Placement - Weekday (Mo = 1)','Placement - Time','Confirmation - Time','Arrival at Pickup - Time','Arrival at Destination - Time','Distance (KM)','Temperature','Rider Id','Time from Pickup to Arrival']].copy()
    new_df.columns = [col.replace(" ","_") for col in new_df.columns] 
    dataset_test  = dataset[['User Id','Platform Type','Personal or Business','Placement - Day of Month','Placement - Weekday (Mo = 1)','Placement - Time','Confirmation - Time','Arrival at Pickup - Time','Arrival at Destination - Time','Distance (KM)','Temperature','Rider Id','Time from Pickup to Arrival']].copy()
    dataset_test.columns = [col.replace(" ","_") for col in dataset_test.columns]
    
    # Encoding personal or business column
    new_df = pd.get_dummies(new_df, columns=['Personal_or_Business'], prefix = ['personal'])
    new_df = new_df.iloc[:, :-1]
    new_df.columns = [*new_df.columns[:-1], 'Business']
    
    # Dealing with the missing values replace with the mean/median temperature
    new_df['Temperature'] = new_df.Temperature.fillna(new_df.Temperature.median())
    dataset_test['Temperature'] = dataset_test.Temperature.fillna(dataset_test.Temperature.median())
    
    # Dealing/ preprocessing time columns
    new_df['Placement_-_Time 24'] = (pd.to_datetime(new_df['Placement_-_Time'].str.strip(), format='%I:%M:%S %p').dt.strftime('%H:%M'))
    new_df['Confirmation_-_Time 24'] = (pd.to_datetime(new_df['Confirmation_-_Time'].str.strip(), format='%I:%M:%S %p').dt.strftime('%H:%M'))
    new_df['Arrival_at_Pickup_-_Time 24'] = (pd.to_datetime(new_df['Arrival_at_Pickup_-_Time'].str.strip(), format='%I:%M:%S %p').dt.strftime('%H:%M'))
    new_df['Arrival_at_Destination_-_Time 24'] = (pd.to_datetime(new_df['Arrival_at_Destination_-_Time'].str.strip(), format='%I:%M:%S %p').dt.strftime('%H:%M'))


    dataset_test['Placement_-_Time 24'] = (pd.to_datetime(new_df['Placement_-_Time'].str.strip(), format='%I:%M:%S %p').dt.strftime('%H:%M'))
    dataset_test['Confirmation_-_Time 24'] = (pd.to_datetime(new_df['Confirmation_-_Time'].str.strip(), format='%I:%M:%S %p').dt.strftime('%H:%M'))
    dataset_test['Arrival_at_Pickup_-_Time 24'] = (pd.to_datetime(new_df['Arrival_at_Pickup_-_Time'].str.strip(), format='%I:%M:%S %p').dt.strftime('%H:%M'))
    dataset_test['Arrival_at_Destination_-_Time 24'] = (pd.to_datetime(new_df['Arrival_at_Destination_-_Time'].str.strip(), format='%I:%M:%S %p').dt.strftime('%H:%M'))
    
    # drop columns
    new_df.drop(['Placement_-_Time', 'Confirmation_-_Time','Arrival_at_Pickup_-_Time','Arrival_at_Destination_-_Time','User_Id','Rider_Id'], axis=1)

    dataset_test.drop(['Placement_-_Time', 'Confirmation_-_Time','Arrival_at_Pickup_-_Time','Arrival_at_Destination_-_Time','User_Id','Rider_Id'], axis=1)
    
    # Spliting data into X and Y , first converting time into hours lapse since the day started
    new_df = new_df[['Platform_Type', 'Arrival_at_Pickup_-_Time 24','Confirmation_-_Time 24','Placement_-_Time 24','Business','Temperature','Distance_(KM)','Placement_-_Weekday_(Mo_=_1)','Placement_-_Day_of_Month','Platform_Type', 'Arrival_at_Destination_-_Time 24', 'Time_from_Pickup_to_Arrival']]
    new_df['Placement_-_Time 24'] = new_df['Placement_-_Time 24'].str.replace(':','.')
    new_df['Confirmation_-_Time 24'] = new_df['Confirmation_-_Time 24'].str.replace(':','.')
    new_df['Arrival_at_Pickup_-_Time 24'] = new_df['Arrival_at_Pickup_-_Time 24'].str.replace(':','.')
    new_df['Arrival_at_Destination_-_Time 24'] = new_df['Arrival_at_Destination_-_Time 24'].str.replace(':','.')

    dataset_test = dataset_test[['Platform_Type', 'Arrival_at_Pickup_-_Time 24','Confirmation_-_Time 24','Placement_-_Time 24','Business','Temperature','Distance_(KM)','Placement_-_Weekday_(Mo_=_1)','Placement_-_Day_of_Month','Platform_Type', 'Arrival_at_Destination_-_Time 24', 'Time_from_Pickup_to_Arrival']]
    dataset_test['Placement_-_Time 24'] = dataset_test['Placement_-_Time 24'].str.replace(':','.')
    dataset_test['Confirmation_-_Time 24'] = dataset_test['Confirmation_-_Time 24'].str.replace(':','.')
    dataset_test['Arrival_at_Pickup_-_Time 24'] = dataset_test['Arrival_at_Pickup_-_Time 24'].str.replace(':','.')
    dataset_test['Arrival_at_Destination_-_Time 24'] = dataset_test['Arrival_at_Destination_-_Time 24'].str.replace(':','.')
    
    x =pd.DataFrame(new_df.iloc[:,:-1].values)
    y = new_df.iloc[:,-1].values
    x_test =pd.DataFrame(new_df.iloc[:,:-1].values)
    y_test= new_df.iloc[:,-1].values
    
    # Rescaling the x values using standardisation
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)
    x_standardise = pd.DataFrame(X_scaled,columns=x.columns)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_scaledtest = scaler.fit_transform(x_test)
    x_standardisetest = pd.DataFrame(x_scaledtest,columns=x.columns)

    
    predict_vector = feature_vector_df[['Pickup Lat','Pickup Long',
                                        'Destination Lat','Destination Long']]
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
