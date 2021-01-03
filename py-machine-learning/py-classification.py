'''
In a machine learning context, classification is a type of supervised learning.

Supervised learning requires that the data fed to the network is already labeled and important features/attributes already separated into distinct categories.

Types of Classification Algorithms:
    1. K-Nearest Neighbour
    2. Support Vector Machines
    3. Decision Trees / Random Forests
    4. Naive Bayes
    5. Linear Discriminant Analysis
    6. Logistic Regression
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt


candidates = {'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
              'admitted': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
              }

df = pd.DataFrame(candidates,columns= ['gmat', 'gpa','work_experience','admitted'])

# Independent Variables
independent_variables = df[['gmat', 'gpa','work_experience']]

# Output/Dependent Variable
output_variable = df['admitted']

#Logistic Regression
'''
    Logistic Regression uses Sigmoid (logit) function to place probabilities on input data, for belonging to a particular class.
    Sigmoid Functions always outputs values between 0 and 1.
    Sigmoid(x) = 1 / (1 + e^-x)
'''
independent_variables_training_dataset, independent_variables_test_dataset, output_variable_training_dataset, output_variable_test_dataset = train_test_split(independent_variables, output_variable, test_size=0.25, random_state=0)

# Create Logistic Regression and train it on training dataset and use it to predict on test dataset.
logistic_regression = LogisticRegression()
logistic_regression.fit(independent_variables_training_dataset, output_variable_training_dataset)
output_variable_predictions = logistic_regression.predict(independent_variables_test_dataset)

# Use Confusion Matrix to determine the accuracy of the Logistic Regression - Classification accuracy
confusion_matrix = pd.crosstab(output_variable_test_dataset, output_variable_predictions, rownames=["Actual"], colnames=["Predicted"])
sn.heatmap(confusion_matrix, annot=True)

print('\nAccuracy:\n', metrics.accuracy_score(output_variable_test_dataset, output_variable_predictions))
plt.show()




