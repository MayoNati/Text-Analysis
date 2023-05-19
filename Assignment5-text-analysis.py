#!/usr/bin/env python
# coding: utf-8

# # Assignment 5 - Text Analysis
# An explanation this assignment could be found in the .pdf explanation document

# 
# ## Materials to review for this assignment
# <h4>From Moodle:</h4> 
# <h5><u>Review the notebooks regarding the following python topics</u>:</h5>
# <div class="alert alert-info">
# &#x2714; <b>Working with strings</b> (tutorial notebook)<br/>
# &#x2714; <b>Text Analysis</b> (tutorial notebook)<br/>
# &#x2714; <b>Hebrew text analysis tools (tokenizer, wordnet)</b> (moodle example)<br/>
# &#x2714; <b>(brief review) All previous notebooks</b><br/>
# </div> 
# <h5><u>Review the presentations regarding the following topics</u>:</h5>
# <div class="alert alert-info">
# &#x2714; <b>Text Analysis</b> (lecture presentation)<br/>
# &#x2714; <b>(brief review) All other presentations</b><br/>
# </div>

# ## Preceding Step - import modules (packages)
# This step is necessary in order to use external modules (packages). <br/>

# In[4]:


# --------------------------------------
import pandas as pd
import numpy as np
# --------------------------------------


# --------------------------------------
# ------------- visualizations:
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# --------------------------------------


# ---------------------------------------
import sklearn
from sklearn import preprocessing, metrics, pipeline, model_selection, feature_extraction 
from sklearn import naive_bayes, linear_model, svm, neural_network, neighbors, tree
from sklearn import decomposition, cluster

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score,classification_report,make_scorer
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# ---------------------------------------


# ----------------- output and visualizations: 
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
# show several prints in one cell. This will allow us to condence every trick in one cell.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# ---------------------------------------


# ### Text analysis and String manipulation imports:

# In[5]:


# --------------------------------------
# --------- Text analysis and Hebrew text analysis imports:
# vectorizers:
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# regular expressions:
import re
# --------------------------------------


# ### (optional) Hebrew text analysis - WordNet (for Hebrew)
# Note: the WordNet is not a must

# #### (optional) Only if you didn't install Wordnet (for Hebrew) use:

# In[6]:


# word net installation:

# unmark if you want to use and need to install
get_ipython().system('pip install wn')
get_ipython().system('python -m wn download omw-he:1.4')


# In[7]:


# word net import:

# unmark if you want to use:
import wn


# ### (optional) Hebrew text analysis - hebrew_tokenizer (Tokenizer for Hebrew)
# Note: the hebrew_tokenizer is not a must

# #### (optional) Only if you didn't install hebrew_tokenizer use:

# In[8]:


# Hebrew tokenizer installation:

# unmark if you want to use and need to install:
get_ipython().system('pip install hebrew_tokenizer')


# In[9]:


# Hebrew tokenizer import:

# unmark if you want to use:
import hebrew_tokenizer as ht


# ### Reading input files
# Reading input files for train annotated corpus (raw text data) corpus and for the test corpus

# In[10]:


train_filename = 'annotated_corpus_for_train.csv'
test_filename  = 'corpus_for_test.csv'
df_train = pd.read_csv(train_filename, index_col=None, encoding='utf-8')
df_test  = pd.read_csv(test_filename, index_col=None, encoding='utf-8')


# In[11]:


df_train.head(8)
df_train.shape


# In[12]:


df_test.head(3)
df_test.shape


# ### Your implementation:
# Write your code solution in the following code-cells

# Student Names: Netanel Mayo.
# 
# 

# In[13]:


# YOUR CODE HERE


# <H1>All steps: </H1>
# <br>
# 
# # Step 1 - Getting to a basic classification:
# **1.0** Change the classification of the species from m => 1 and f=> 0 in train data.
# 
# **1.1** Differences between 2 types of Vectorizers with only max_features = 10000 and 6 models by simply training without playing with any parameters tuning by using cross validation
# 
# **Models**
# 1. **KNN**
# 2. **DecisionTree** 
# 3. **LinearSVC** 
# 4. **Perceptron** 
# 5. **SGDClassifier**
# 6. **MultinomialNB**
# 
# **Vectorizers**
# 1. **tfidf Vectorizer**
# 2. **Count Vectorizer**
# 
# # Step 2 - Ability to measure system performance:
# **2.0** Finding the best hyperparameters for the model and for Vectorizers, so we will run GridSearchCV <br>
# It takes time as you let it try more options of hyperparameters. <br>
# 
# **2.1** After we have found the combination that will give us the best value by using GridSearchCV,<br> 
# we will run the cross-validation again this time with the new parameters.<br> 
# Cmpare it against the cross-validation without tuning the parameters.<br>
# 
# **2.2** We will see in the bar chart that by finding the best parameters<br> 
# for the most part the result of the f1 average will increase (we will also see the change in percentages).<br>
# 
# 
# # Step 3 - Evaluate performance:
# 
# **3.0** In this step, before we start predicting our test data, we will want to split the training data by train_test_split in order to evaluate each model's performance.<br>
# A comparison will be made between models that haven't been tuning parameters and models that have been tuning parameters to see how the cleanliness before training the model affects the model's performance.<br> 
# The results of all the tests will be summarized in a table based on the F1 average, showing the results of all the tests.<br>
# 
# **3.1** We will see two bar charts in 2 of them we will see models that underwent parameter tuning vs. models that did not undergo parameter tuning:
# 
# 1. Preprocessed models before training by cleaning numbers,punctuation,spaces,English word.
# 2. Models that have not undergone any pre-processing
# 
# 
# **3.2** We will see a summary table <br>
# 
# 
# # Step 4 - Predictive out data test:
#     
# **4.0** Will automatically select the model that produced the highest f1 average value and use it to predict our test data<br>
# 
# **4.1** Predict our test data<br>
# 
#     
# 
#     
# 
# 
# 
# 

# # step 1 - Getting to a basic classification:

# **1.0** 
# 

# In[14]:


df_train['gender'] = df_train['gender'].map({'f': 0, 'm': 1})
df_train


# In[481]:


def clean_text(text):
    # Remove numbers , spaces and leading and trailing spaces
    text = clean_english_words(text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)#remove punctuation
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


# In[482]:


def clean_english_words(text):
    englishWords=[]
    tokens = ht.tokenize(text)
    for grp, token, token_num, (start_index, end_index) in tokens:
        if grp == 'ENGLISH':
            englishWords.append(token)
            text = text.replace(token, "")
            text = text.strip()
    return text


# In[483]:


def VectorizerTypes(vectorizer,X_train,y_train):
    
    # Convert the training data to a list of lists.
    data_train_list = X_train.tolist()

    # Fit the vectorizer to the training data.
    vectorizer.fit(data_train_list)

    # Transform the training data using the vectorizer.
    df_train_Vectorizer = vectorizer.transform(data_train_list)

    # Convert the vectorized data to a pandas DataFrame.
    df_train_Vectorizer = pd.DataFrame(df_train_Vectorizer.toarray(), columns=vectorizer.get_feature_names_out())

    # Add the labels to the DataFrame.
    df_train_Vectorizer['gender'] = y_train

    # Return the vectorized training data.
    return df_train_Vectorizer


# In[488]:


import time

def CrossValidation(dataset,model):
      
    # Start the timer.

    start_time = time.time()

    # Get the features and labels.

    X = dataset[dataset.columns[(dataset.columns != 'gender')]]
    y_gender = dataset['gender']

    # Create a scorer for F1 score for male.

    f1_male_scorer = make_scorer(f1_score, pos_label=1)

    # Calculate the F1 score for male using 10-fold cross-validation.

    f1_male = cross_val_score(model, X, y_gender, cv=10, scoring=f1_male_scorer, n_jobs=10).mean()

    # Create a scorer for F1 score for female.

    f1_female_scorer = make_scorer(f1_score, pos_label=0)

    # Calculate the F1 score for female using 10-fold cross-validation.

    f1_female = cross_val_score(model, X, y_gender, cv=10, scoring=f1_female_scorer, n_jobs=10).mean()

    # Calculate the average F1 score.

    f1_avg = (f1_male + f1_female) / 2

    # Stop the timer and calculate the elapsed time.

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Return the F1 scores and the elapsed time.

    return f1_male, f1_female, f1_avg, elapsed_time


# In[489]:


#  This function performs 10-fold cross-validation on the given dataset using the given vectorizers and models.
def performance_test_by_cross_validation(Vectorizers_types_dict, models_dict, X_train,y_train):
    
    models = list(models_dict.keys())

    # Create a dictionary that maps vectorizer types to a list of DataFrames containing the F1 scores for male, female, and average, and the elapsed time.

    VectorizerTypes_diff_model = {}

    # Create a DataFrame that summarizes the F1 scores for each vectorizer and model.

    summarizes_df = pd.DataFrame(index=Vectorizers_types_dict.keys(), columns=models_dict.keys())

    # Iterate over the vectorizer types.
    for key_Vec, value_Vec in Vectorizers_types_dict.items():
        # Get the vectorizer type.

        vectorizer_type = key_Vec

        # Create a vectorizer.

        vectorizer = Vectorizers_types_dict[vectorizer_type]["func"](**Vectorizers_types_dict[vectorizer_type]["params"])

        # Create a pipeline that performs 10-fold cross-validation on the vectorizer and the given models.

        pip_modelsCrossValidation = Pipeline(steps=[])

        # Iterate over the model types.
        for key_alg, value_alg in models_dict.items():
            
            # Print the vectorizer type and the model type.
            print(f"{key_Vec} and {key_alg}")

            # Perform 10-fold cross-validation on the vectorizer and the model.

            crossValidation = CrossValidation(VectorizerTypes(vectorizer, X_train, y_train), models_dict[key_alg])

            # Add the cross-validation results to the pipeline.

            pip_modelsCrossValidation.steps.append((key_alg, crossValidation))

        CrossValidation_diff_df = pd.DataFrame(index=["f1_male", "f1_female", "f1_avg", "elapsed_time"], columns=models)
        
        for model in models:
            alg = str(model)
            if alg in pip_modelsCrossValidation.named_steps:
                f1_male, f1_female, f1_avg, elapsed_time = pip_modelsCrossValidation[alg]
                CrossValidation_diff_df.loc['f1_male', alg] = f1_male
                CrossValidation_diff_df.loc['f1_female', alg] = f1_female
                CrossValidation_diff_df.loc['f1_avg', alg] = f1_avg
                CrossValidation_diff_df.loc['elapsed_time', alg] = elapsed_time
                summarizes_df.loc[vectorizer_type, alg] = f1_avg
                
        VectorizerTypes_diff_model[vectorizer_type] = [CrossValidation_diff_df,pip_modelsCrossValidation]

    return VectorizerTypes_diff_model,summarizes_df


# **1.1**

# In[490]:


max_features = 10000
tfidf_Vectorizer_params = {"max_features": max_features,}
Count_Vectorizer_params = {"max_features": max_features,}

Vectorizers_types_dic = {
    "tfidf": {"func": TfidfVectorizer, "params": tfidf_Vectorizer_params},
    "vect": {"func": CountVectorizer, "params": Count_Vectorizer_params}
}
models = {
        "KNN": KNeighborsClassifier(),       
        "LinearSVC": LinearSVC(),
        "Perceptron": Perceptron(),
        "SGDClassifier": SGDClassifier(),
        "MultinomialNB": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
}



# In[491]:


basic_crossVal_df=pd.DataFrame()
basic_crossVal_dict,basic_crossVal_df=performance_test_by_cross_validation(Vectorizers_types_dic, models,df_train['story'],df_train['gender'])


# This study examines the differences between two types of vectorizers, with a maximum of 10,000 features, and evaluates six machine learning models using cross-validation without any parameter tuning.
# 

# In[578]:


basic_crossVal_df


# **A graph depicting the average F1 scores for each model based on different vectorization methods.**

# In[582]:


#Create a bar chart from the DataFrame
def models_comparison_graph(df):    
    fig, ax = plt.subplots(figsize=(12, 8))
    df.plot(kind='bar', ax=ax, width=0.9)
    # Set the chart title and axis labels
    ax.set_title(' f1 average\Vectorizers and models')
    ax.set_xlabel('Vectorizers')
    ax.set_ylabel('f1 average')
    ax.grid(True)
    ax.set_ylim([0, 1])
    ax.set_yticks([i/10 for i in range(11)])
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    # add column name labels to the bars
    for i, bar in enumerate(ax.containers):
        for j, rect in enumerate(bar.patches):
            x = rect.get_x() + rect.get_width() / 2
            y = rect.get_height()

            # add cell value labels to the bars
            value = df.iloc[j, i % 10]
            ax.text(x, y - 0.1, f'{value:.3f}', ha='center', va='bottom')
            plt.rcParams['font.size'] = 11


    plt.show()



# In[583]:


models_comparison_graph(basic_crossVal_df)


# # Step 2 - Ability to measure system performance:

# This function performs a grid search to find the optimal hyperparameters for a given set of machine learning models, with different vectorization techniques. It takes as input the models, hyperparameters, vectorizers, vectorizer parameters, and training data.
# 
# For each model and vectorizer combination, the function initializes a pipeline, performs a grid search using the specified hyperparameters and F1 macro scoring metric, and stores the results in a DataFrame and two dictionaries. Finally, the function returns the stored results and prints the total elapsed time for the grid search.

# In[495]:


def grid_search_model_vectorizer(models, hyperparameters, vectorizers, Vectorizerparameters, X_train, y_train):
    X = X_train
    y = y_train
    
    df_total_scors_combination = pd.DataFrame(index=vectorizers.keys(), columns=models.keys())
    combination_best_param_dict = {}
    combination_all_dict = {}

    start_time = time.time()
    
    # Perform grid search for each model and hyperparameter combination
    for model_name, model in models.items():
        start_time_model = time.time()
        if model_name in hyperparameters:
            parameters = hyperparameters[model_name]
            vector_dict = {}
            best_score = 0
            for vectorizer_name, vectorparameters in Vectorizerparameters.items():
                

                if vectorizer_name in Vectorizerparameters:
                    start_time_vector = time.time()
                    
                    vectorizer_parameters_dict = {}

                    paramList = {}
                    pipeline = Pipeline(steps=[])
                    print("----")
                    print("Start")
                    print("Tuning {} and {}...".format(vectorizer_name,model_name))

                    for parameter_name, parameters_value in vectorparameters.items():
                        paramList[f'{vectorizer_name}__{parameter_name}'] = parameters_value

                    for keyItems, valueItems in parameters.items():
                        paramList[f'{model_name}__{keyItems}'] = valueItems

                    pipeline.steps.append((vectorizer_name, vectorizers[vectorizer_name]))
                    pipeline.steps.append((model_name, models[model_name]))

                    grid_search = GridSearchCV(pipeline, paramList, cv=5, scoring="f1_macro",n_jobs=10)
                    grid_search.fit(X, y)               
                    best_model = grid_search.best_estimator_
    
                    print("Best parameters: {}".format(grid_search.best_params_))
                    print("Best score: {:.3f}\n".format(grid_search.best_score_))
                    
                    
                    vectorizer_parameters_dict[vectorizer_name] = grid_search.best_params_
                    vector_dict[vectorizer_name] = best_model

                    if grid_search.best_score_ > best_score:
                        best_score = grid_search.best_score_
                        combination_best_param_dict[model_name] = vectorizer_parameters_dict
                        
                        
                    df_total_scors_combination.loc[vectorizer_name, model_name] = grid_search.best_score_
                    end_time_vec_model = time.time()
                    elapsed_time_vec_model = end_time_vec_model - start_time_vector
                    print(f"Time elapsed {model_name} and {vectorizer_name}: {elapsed_time_vec_model}")

                    print("End")
                    print("----")

            combination_all_dict[model_name]=vector_dict
            end_time_total_model = time.time()
            elapsed_time_total_model = end_time_total_model - start_time_model
            print(f"Total time for {model_name} : {elapsed_time_total_model}")
            
    end_time_total_all_models = time.time()
    elapsed_time_total_all_models = end_time_total_all_models - start_time
    print(f"Total time for all models: {elapsed_time_total_all_models}")

    return combination_all_dict,combination_best_param_dict,df_total_scors_combination


# In[496]:


hyperparameters = {

    "KNN": {"n_neighbors": [3, 5],
            "weights": ['uniform'],    
            'algorithm': ['auto'],
            'leaf_size': [30],
           'n_jobs':[10]},

    "Decision Tree": {'criterion': ['gini'],
                      'max_depth': [None, 10],
                      'min_samples_split': [5, 10],
                      'min_samples_leaf': [1, 2]},

    "LinearSVC": {"C": [0.1, 1, 10], 
                  "penalty": ["l2"], 
                  "loss": ['squared_hinge'],
                  "fit_intercept": [True],
                  'dual': [True, False]},


         "Perceptron": {"alpha": [0.0001, 0.001, 0.01,0.05], 
                    "penalty": ['elasticnet'],
                    'max_iter': [1000],
                   'n_jobs':[10]},
    
    
    "SGDClassifier": {"alpha": [0.0001, 0.001, 0.01,0.05, 0.1], 
                      "loss": ['hinge', 'log'], 
                      'max_iter': [2000]},

    "MultinomialNB": {"alpha": [0.1, 0.5], "fit_prior": [True, False]},

}


vectorizers = {"tfidf": TfidfVectorizer(),"vect": CountVectorizer()}

Vectorizerparameters = {
    "tfidf": {"min_df": [1, 3,7],"max_df": [0.3,0.7, 0.9],"ngram_range": [(1,1), (1,2)],"max_features": [10_000,30_000, 40_000], "sublinear_tf": [True]},
     "vect": {"min_df": [1, 3,9],"max_df": [0.7, 0.9],"ngram_range": [(1,1), (1,2)],"max_features": [10_000,30_000, 40_000]},
}




# **2.0**

# In[500]:


df_train_clean=pd.DataFrame()
df_train_clean['story']=df_train['story'].copy()
df_train_clean['story']=df_train_clean['story'].apply(clean_text)

X = df_train_clean['story']
y = df_train['gender']
dict_all,dict_best_param,df_total_scors_combination=grid_search_model_vectorizer(models, hyperparameters, vectorizers, Vectorizerparameters, X, y)



# In[584]:


df_total_scors_combination


# These are the results of the best combinations for the models

# In[585]:


models_comparison_graph(df_total_scors_combination)


# **2.1**

# In[506]:


df_train_temp['story']=df_train['story'].copy()
df_train_temp['story']=df_train_temp['story'].apply(clean_text)


crossValidation_after_best_param=pd.DataFrame(index=vectorizers.keys(), columns=models.keys())
for model_name in list(dict_all.keys()):
    for vector_name in list(Vectorizers_types_dic.keys()):  
        print(f"{model_name} - {vector_name}")
        f1_male,f1_female,f1_avg,elapsed_time=CrossValidation(
            VectorizerTypes(dict_all[model_name][vector_name][0], df_train_clean['story'],df_train['gender']),
            dict_all[model_name][vector_name][1])
        crossValidation_after_best_param.loc[vector_name, model_name] = f1_avg

crossValidation_after_best_param


# In[507]:


print("This table displays the performance of all the models with their respective vectorizers, without any parameter tuning. The results are presented as the F1 score average.")
basic_crossVal_df

print("This table displays the best results obtained for each model by tuning specific parameters, along with the corresponding vectorizer that achieved the highest results. The results are presented as the F1 score average.")
crossValidation_after_best_param


# **We will now view a graph that illustrates the differences between models trained with their respective vectors without any adjustments, and models for which we found the best parameters that resulted in the highest F1 score average.**

# In[509]:


import seaborn as sns


print("-------------------------------------TfidfVectorizer------------------------------------------")
tfidf_pd = pd.concat([basic_crossVal_df.iloc[0:1], crossValidation_after_best_param.iloc[0:1], ])
tfidf_pd = tfidf_pd.reset_index(drop=True)
tfidf_pd = tfidf_pd.rename(index={0: 'Cross_val_Before_Tuning', 1: 'Cross_val_After_Tuning'})
tfidf_pd
print("----------------------------------------------------------------------------------------------")

print("-------------------------------------CountVectorizer------------------------------------------")
vect_pd = pd.concat([basic_crossVal_df.iloc[1:2], crossValidation_after_best_param.iloc[1:2], pd.DataFrame( )])
vect_pd = vect_pd.reset_index(drop=True)
vect_pd = vect_pd.rename(index={0: 'Cross_val_Before_Tuning', 1: 'Cross_val_After_Tuning'})
vect_pd
print("----------------------------------------------------------------------------------------------")
# display the new dataframe


# In the lower graphs, we will see two plots: the first one depicts models with Tfidf vectorizers, while the second one shows models with Count vectorizers. The plots compare the results of training models with default values versus those obtained after finding the best parameters, based on the highest f1 average score.

# If we see a bar that is colored **red**, it means that we did not observe any performance improvement between the models trained with default values and the ones trained after finding the best parameters
# 

# In[596]:


import matplotlib.pyplot as plt

def results_comparison_graph(df1,text_df1,text_lagend_df1,df2,text_df2,text_lagend_df2):
    crossval_sum_df_copy = pd.DataFrame()
    crossval_sum_df_copy = CrossVal_summarizes_df
    gridSearchCV_best_p_df_copy = pd.DataFrame()
    gridSearchCV_best_p_df_copy = df_total_scors_combination

    # Create a figure and axis for the plot
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 12))

    # Define the x-axis tick labels
    tick_labels = crossval_sum_df_copy.columns

    # Set the bar width
    bar_width = 0.35
    # Loop through each pair of cells with the same location in the old and new DataFrames
    for i, (old_val, new_val) in enumerate(zip(df1.values.flatten(), df2.values.flatten())):
        # Compute the x-position of the bar for this cell
        x_pos = i % len(tick_labels)

        # Compute the y-position of the bar for this cell
        y_pos = i // len(tick_labels)

        if new_val < old_val:
            color_old_val = color_new_val = 'red'
        else:
            color_new_val = 'blue'
            color_old_val = 'orange'

        # Create a bar for this cell in the old DataFrame
        ax = ax1 if y_pos == 0 else ax2
        ax.bar(x_pos - bar_width / 2, old_val, bar_width, color=color_new_val)

        # Create a bar for this cell in the new DataFrame
        ax.bar(x_pos + bar_width / 2, new_val, bar_width, color=color_old_val)

        # Add the cell value as a text annotation to each bar
        ax.text(x_pos - bar_width / 2, old_val + 0.1, f'{old_val:.3f}', ha='center', va='bottom')
        ax.text(x_pos + bar_width / 2, new_val + 0.1, f'{new_val:.3f}', ha='center', va='bottom')


        x_label = "Models"
        y_label = "f1 avg score"
        # Set the chart title and axis labels for each subplot
        ax1.set_title(text_df1)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)

        ax2.set_title(text_df2)
        ax2.set_xlabel(x_label)
        ax2.set_ylabel(y_label)

        # Set the x-axis tick labels and limits for each subplot
        ax.set_xticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels)
        ax.set_xlim([-bar_width, len(tick_labels) - bar_width])

        # Set the y-axis limits and tick marks for each subplot
        ax.set_ylim([0, max(df1.values.max(), df2.values.max()) + 1])
        ax.set_yticks(range(int(ax.get_ylim()[1]) + 1))
        ax.set_ylim([0, 1])

        ax.grid(True)

        # Add a legend to the first subplot
        if y_pos == 0:
            ax.legend([text_lagend_df1, text_lagend_df2], loc='lower right')

    # Show the plot
    plt.show()



# **2.2**

# In[597]:


text_df1=f'Comparison of training models with default parameters and models with optimized parameters based on their F1 average scores ({basic_crossVal_df.iloc[0].name})'
text_df2=f'Comparison of training models with default parameters and models with optimized parameters based on their F1 average scores ({basic_crossVal_df.iloc[0].name})'
results_comparison_graph(basic_crossVal_df,text_df1,"Befor tuning parameters cross validation",crossValidation_after_best_param,text_df2,"After tuning parameters cross validation")


# In[515]:


import seaborn as sns
import matplotlib.pyplot as plt

def results_comparison_graph_precent(df1,df2):
    pct_change = (( df2 / df1 ) - 1) * 100
    pct_change
    # convert the data to float dtype
    pct_change = pct_change.astype(float)
    sns.heatmap(pct_change, annot=True, cmap='coolwarm')
    plt.title('Percentage Change between models and vectorizers without tuning parameters and models with tuning parameters')
    plt.show()


# In[516]:


results_comparison_graph_precent(basic_crossVal_df,crossValidation_after_best_param)


# # Step 3 - Evaluate performance:

# In[525]:


#This function fits a model to the training data, predicts the gender for the test data, 
#and calculates the F1 score and accuracy.

def train_and_evaluate_model(model_name, vectorizer, X_train, X_test, y_train, y_test):
    
    X_train_list=X_train.tolist()
    vectorizer.fit(X_train_list)
    X_train_vec = vectorizer.transform(X_train_list)
    X_test_vec = vectorizer.transform(X_test.tolist())
    model_trained = model_name.fit(X_train_vec, y_train)  # fitting our model with the train values
    y_pred = model_trained.predict(X_test_vec)
    y_pred_train = model_trained.predict(X_train_vec)
    
    # Calculate F1 score for male class (1)
    f1_male = f1_score(y_test, y_pred, pos_label=1)
    # Calculate F1 score for female class (0)
    f1_female = f1_score(y_test, y_pred, pos_label=0)
    # Calculate overall average weighted F1 score
    f1_average_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_avg = (f1_male + f1_female) / 2
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    
    # Return the fitted model, the F1 score, and the accuracy.
    return model_trained, f1_avg,f1_average_weighted ,accuracy


# We will perform another comparison, this time on the actual prediction performance of our training data. The comparison will be between the default models and the models trained with the best parameters based on the f1 average score

# In[544]:


summrize_table=pd.DataFrame(columns=['models', 'Vector type', 'f1 ave','f1 ave weighted','Accuracy','params','preprocessing'])


# **3.0**

# **With clean the text before**

# In[545]:


df_train_temp = pd.read_csv(train_filename, index_col=None, encoding='utf-8')
df_test_temp  = pd.read_csv(test_filename, index_col=None, encoding='utf-8')
df_train_temp['gender'] = df_train_temp['gender'].map({'f': 0, 'm': 1})
df_train_temp['story']=df_train_temp['story'].apply(clean_text)
X_train_split_temp, X_test_split_temp, y_train_split_temp, y_test_split_temp = train_test_split(df_train_temp['story'], df_train_temp['gender'], test_size=0.2, random_state=0)

basic_model_predict_evaluate_clean_df = pd.DataFrame(index=Vectorizers_types_dic.keys(),columns=models.keys())
pipeline = Pipeline(steps=[])

for vector_name in list(Vectorizers_types_dic.keys()):  
    vectorizer_pre=Vectorizers_types_dic[vector_name]["func"](**Vectorizers_types_dic[vector_name]["params"])
    
    for model_name in list(models.keys()):
        model_trained, f1_avg,f1_average_weighted,accuracy = train_and_evaluate_model(models[model_name],vectorizer_pre,X_train_split_temp,X_test_split_temp,y_train_split_temp,y_test_split_temp)
        basic_model_predict_evaluate_clean_df.loc[vector_name, model_name] = f1_avg
        
        pipeline.steps.append((vector_name, vectorizer_pre))
        pipeline.steps.append((model_name, models[model_name]))
        new_row = {'models': model_name, 
                   'Vector type': vector_name, 
                   'f1 ave': f1_avg,
                   'f1 ave weighted':f1_average_weighted,
                   'Accuracy':accuracy,
                   'params':pipeline,
                   'preprocessing':"clean text"}
        summrize_table.loc[len(summrize_table)] = new_row

        
basic_model_predict_evaluate_clean_df

X_train_split_temp, X_test_split_temp, y_train_split_temp, y_test_split_temp = train_test_split(df_train_temp['story'], df_train_temp['gender'], test_size=0.2, random_state=0)
after_tuning_param_clean_df = pd.DataFrame(index=Vectorizers_types_dic.keys(),columns=models.keys())
for model_name in list(dict_all.keys()):
    for vector_name in list(Vectorizers_types_dic.keys()):  
        model_trained, f1_avg,f1_average_weighted,accuracy = train_and_evaluate_model(dict_all[model_name][vector_name][1],dict_all[model_name][vector_name][0],X_train_split_temp,X_test_split_temp,y_train_split_temp,y_test_split_temp)        
        after_tuning_param_clean_df.loc[vector_name, model_name] = f1_avg
        new_row = {'models': model_name, 
                   'Vector type': vector_name, 
                   'f1 ave': f1_avg,
                   'f1 ave weighted':f1_average_weighted,
                   'Accuracy':accuracy,
                   'params':dict_all[model_name][vector_name],
                   'preprocessing':"clean text"}
        summrize_table.loc[len(summrize_table)] = new_row

        
after_tuning_param_clean_df


# **3.1**

# In[598]:


text_df1=f'After clean data comparison of training models without tuning parameters of the models, optimized parameters based on their F1 average scores ({CrossVal_summarizes_df.iloc[0].name})'
text_df2=f'After clean data comparison of training models with tuning parameters of the models, optimized parameters based on their F1 average scores ({CrossVal_summarizes_df.iloc[1].name})'

results_comparison_graph(basic_model_predict_evaluate_clean_df,text_df1,"Clean Text, Before tuning params",after_tuning_param_clean_df,text_df2,"Clean Text, After tuning params")


# **Without clean the text before**

# In[547]:


df_train_temp = pd.read_csv(train_filename, index_col=None, encoding='utf-8')
df_test_temp  = pd.read_csv(test_filename, index_col=None, encoding='utf-8')
df_train_temp['gender'] = df_train_temp['gender'].map({'f': 0, 'm': 1})

print("---------------Basic param evaluate--------------------")
X_train_split_temp, X_test_split_temp, y_train_split_temp, y_test_split_temp = train_test_split(df_train_temp['story'], df_train_temp['gender'], test_size=0.2, random_state=42)
basic_model_predict_evaluate_df = pd.DataFrame(index=Vectorizers_types_dic.keys(),columns=models.keys())
for vector_name in list(Vectorizers_types_dic.keys()):  
    vectorizer_pre=Vectorizers_types_dic[vector_name]["func"](**Vectorizers_types_dic[vector_name]["params"])
    
    for model_name in list(models.keys()):
        model_trained, f1_avg,f1_average_weighted,accuracy = train_and_evaluate_model(models[model_name],vectorizer_pre,X_train_split_temp,X_test_split_temp,y_train_split_temp,y_test_split_temp)
        basic_model_predict_evaluate_df.loc[vector_name, model_name] = f1_avg
        new_row = {'models': model_name, 
                   'Vector type': vector_name, 
                   'f1 ave': f1_avg,
                   'f1 ave weighted':f1_average_weighted,
                   'Accuracy':accuracy,
                   'params':pipeline,
                   'preprocessing':"nothing"}
        summrize_table.loc[len(summrize_table)] = new_row
        

        
basic_model_predict_evaluate_df


print("------------------------------------------------------")

print("---------------Best param evaluate--------------------")


X_train_split_temp, X_test_split_temp, y_train_split_temp, y_test_split_temp = train_test_split(df_train_temp['story'], df_train_temp['gender'], test_size=0.2, random_state=42)
after_tuning_param_df = pd.DataFrame(index=Vectorizers_types_dic.keys(),columns=models.keys())
for model_name in list(dict_all.keys()):
    for vector_name in list(Vectorizers_types_dic.keys()):  
        model_trained, f1_avg,f1_average_weighted,accuracy = train_and_evaluate_model(dict_all[model_name][vector_name][1],dict_all[model_name][vector_name][0],X_train_split_temp,X_test_split_temp,y_train_split_temp,y_test_split_temp)        
        after_tuning_param_df.loc[vector_name, model_name] = f1_avg
        new_row = {'models': model_name, 
                   'Vector type': vector_name, 
                   'f1 ave': f1_avg,
                   'f1 ave weighted':f1_average_weighted,
                   'Accuracy':accuracy,
                   'params':dict_all[model_name][vector_name],
                   'preprocessing':"nothing"}
        summrize_table.loc[len(summrize_table)] = new_row
after_tuning_param_df
print("------------------------------------------------------")


# **3.1**

# In[599]:


text_df1=f'Without clean data comparison of training models without tuning parameters of the models, optimized parameters based on their F1 average scores ({CrossVal_summarizes_df.iloc[0].name})'
text_df2=f'Without clean data comparison of training models with tuning parameters of the models, optimized parameters based on their F1 average scores ({CrossVal_summarizes_df.iloc[1].name})'

results_comparison_graph(basic_model_predict_evaluate_df,text_df1,"Without clean text, Before tuning params",after_tuning_param_df,text_df2,"Without clean text, After tuning params")


# **3.2**

# In[563]:


summrize_table = summrize_table.sort_values(by='f1 ave', ascending=False)
summrize_table = summrize_table.reset_index(drop=True)
summrize_table = summrize_table.drop(['index', 'level_0'], axis=1)
summrize_table


# # Step 4 - Predictive out data test:
# 

# **4.0**

# In[570]:


max_value_row = summrize_table[summrize_table['f1 ave'] == summrize_table['f1 ave'].max()]
max_value_row

model_select=(max_value_row["params"].values)[0]
model_select


# In[571]:


def predict_gender(model_selected,Vectorizer_selected,df_train,df_test):
    df_test_copy=pd.DataFrame()
    df_test_copy=df_test.copy()
    df_test_copy['story'] = df_test_copy['story'].apply(clean_text)

    df_train_copy=pd.DataFrame()
    df_train_copy=df_train.copy()
    df_train_copy['story'] = df_train_copy['story'].apply(clean_text)
    
    X_train = df_train_copy['story']
    y_train = df_train_copy['gender']
    X_test = df_test_copy['story']
     
    X_train_list=X_train.tolist()
    Vectorizer_selected.fit(X_train_list)
    X_train_vec = Vectorizer_selected.transform(X_train_list)
    X_test_vec = Vectorizer_selected.transform(X_test.tolist())
    
    model_trained = model_selected.fit(X_train_vec, y_train)  # fitting our model with the train values
    y_pred = model_trained.predict(X_test_vec)
    
    df_test_predicted=pd.DataFrame()
    df_test_predicted['test_example_id']=df_test['test_example_id']
    df_test_predicted['predicted_category']=y_pred
    df_test_predicted['predicted_category'] = df_test_predicted['predicted_category'].map({0: 'f', 1: 'm'})
    
    return df_test_predicted


# **4.1**

# In[576]:


df_predicted = pd.DataFrame()
df_predicted=predict_gender(model_select[1],model_select[0],df_train,df_test)
pd.concat([df_predicted.head(5), df_predicted.tail(5)])


# ### Save output to csv (optional)
# After you're done save your output to the 'classification_results.csv' csv file.<br/>
# We assume that the dataframe with your results contain the following columns:
# * column 1 (left column): 'test_example_id'  - the same id associated to each of the test stories to be predicted.
# * column 2 (right column): 'predicted_category' - the predicted gender value for each of the associated story. 
# 
# Assuming your predicted values are in the `df_predicted` dataframe, you should save you're results as following:

# In[577]:


df_predicted.to_csv('classification_results.csv',index=False)

