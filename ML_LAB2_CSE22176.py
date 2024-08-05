#Q1
import pandas as pd
import numpy as nm


data=pd.read_excel(r"Lab Session Data.xlsx", sheet_name='Purchase data')

# data=pd.read_excel(sheet, 'Purchase data')
A=['Candies (#)','Mangoes (Kg)','Milk Packets (#)']
a=pd.read_excel(r"Lab Session Data.xlsx",usecols=A)
C=['Payment (Rs)']
c=pd.read_excel(r"Lab Session Data.xlsx",usecols=C)

dimen=a.shape[1]
vec_count=a.shape[0]
rank=nm.linalg.matrix_rank(a)
pseudoinv=nm.linalg.pinv(a)

print("The dimensionality is", dimen)
print("The numbers of vs is", vec_count)
print("The rank is", rank)
#Q2
price=nm.dot(pseudoinv,c)
print("The Cost of each product available is", price)
#Q3

import pandas as pd
import numpy as nm


data=pd.read_excel(r"Lab Session Data.xlsx", sheet_name='Purchase data')
data['status'] = data['Payment (Rs)'].apply(lambda x: "RICH" if x > 200 else "POOR")
print(data[['Payment (Rs)', 'status']].head())

#Q4
import statistics
import pandas as pd
import matplotlib.pyplot as plot

data=pd.read_excel(r"Lab Session Data.xlsx", sheet_name='IRCTC Stock Price')

pricedata=data['Price']
daily_mean=statistics.mean(pricedata)
print("The mean of price is ",daily_mean)
print("The variance in price is ",statistics.variance(pricedata))

wed_data=data[data['Day']=='Wed']
mean_wed=statistics.mean(wed_data['Price'])
print("The mean price on wednesdays is ",mean_wed)
print("The difference between the mean price usually and mean price wednesday's is ",daily_mean-mean_wed)

apr_data = data[data['Month']=='Apr']
mean_apr=statistics.mean(apr_data['Price'])
print("The mean price in april is ",mean_apr)
print("The difference between the mean price usually and mean price in april is ",daily_mean-mean_apr)

"""chgdata=data['Chg%']
prob_loss = len(chgdata[chgdata < 0]) / len(chgdata)
print("The probability of loss is ",prob_loss)"""

prob_profit_wednesday = len(wed_data[wed_data['Chg%'] > 0]) / len(wed_data)
print("Probability of making a profit on Wednesday", prob_profit_wednesday)

inv_denom=(len(wed_data) / len(data))
conditional_prob = prob_profit_wednesday *inv_denom
print("The probability of profit given it is wedneday is ",conditional_prob)


plot.scatter(data['Day'], data['Chg%'])
plot.xlabel('Day of the Week')
plot.ylabel('Chg%')
plot.title('Chg% vs Day of the Week')
plot.show()
#Q5

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_excel(r"Lab Session Data.xlsx", sheet_name='thyroid0387_UCI')
print(data.dtypes)
cat_cols = data.select_dtypes(include=['object']).columns
num_cols = data.select_dtypes(include=['int64']).columns

for i in cat_cols:
    values=data[i].unique()
    print("Unique values in column",i," ",values)

    if len(values) < 100:
            print(f"Suggested Encoding: Label Encoding")
            data[i] = data[i].astype('category').cat.codes
    else:
        print(f"Suggested Encoding: One-Hot Encoding")
        data = pd.get_dummies(data, columns=[i], prefix=[i])

for i in num_cols:
    print("Data range for numeric attributes such as", i," is ",data[i].min()," to ",data[i].max())

data.replace('?', np.nan, inplace=True)

print("Number of missing values in each feature")
missing_values = data.isnull().sum()
print(missing_values)

print("\nOutliers in Numeric Data:")
for i in num_cols:
    sns.boxplot(x=data[i])
    plt.title("Boxplot for {}".format(i))
    plt.show()


for i in num_cols:
    print("Mean and std. deviation for", i, "is ", data[i].mean()," and ", data[i].std())
#Q6

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_excel(r"Lab Session Data.xlsx", sheet_name='thyroid0387_UCI')
data.replace('?', np.nan, inplace=True)

num_cols = data.select_dtypes(include=['int64']).columns

for i in num_cols:
    if data[i].isnull().sum() > 0:
        if data[i].skew() < 1:
            data[i].fillna(data[i].mean(), inplace=True)
            print("Column: {}, Imputation Method: Mean".format(i))
        else:
            data[i].fillna(data[i].median(), inplace=True)
            print("Column: {}, Imputation Method: Median".format(i))

for i in cat_cols:
    if data[i].isnull().sum() > 0:
        data[i].fillna(data[i].mode()[0], inplace=True)
        print("Column: {}, Imputation Method: Mode".format(i))
pip install scikit-learn

#Q7

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data=pd.read_excel(r"Lab Session Data.xlsx", sheet_name='thyroid0387_UCI')
print(data[num_cols].describe())
#high standard deviation for age
scaler = StandardScaler()
data['age'] = scaler.fit_transform(data[['age']])

print("Z-score Normalized 'age' Column:")
print(data['age'])
#Q8
import pandas as pd
import numpy as np


data=pd.read_excel(r"Lab Session Data.xlsx", sheet_name='thyroid0387_UCI')
data.replace('?', np.nan, inplace=True)
print("For thyroid sheet")

binary_cols = [col for col in data.columns if set(data[col].dropna().unique()) <= {'t', 'f'}]
binary_cols += [col for col in data.columns if set(data[col].dropna().unique()) <= {'M', 'F'}]

data[binary_cols] = data[binary_cols].replace({'t': 1, 'f': 0, 'M': 1, 'F': 0})

v1 = data.loc[0, binary_cols].astype(int)
v2 = data.loc[1, binary_cols].astype(int)


f11 = np.sum((v1 == 1) & (v2 == 1))
f00 = np.sum((v1 == 0) & (v2 == 0))
f10 = np.sum((v1 == 1) & (v2 == 0))
f01 = np.sum((v1 == 0) & (v2 == 1))

jc = f11 / (f01 + f10 + f11)
smc = (f11 + f00) / (f00 + f01 + f10 + f11)

print("Jaccard Coefficient (JC):", jc)
print("Simple Matching Coefficient (SMC):", smc)

data=pd.read_excel(r"Lab Session Data.xlsx", sheet_name='marketing_campaign')
data.replace('?', np.nan, inplace=True)
print("For marketing sheet")

binary_cols = [col for col in data.columns if set(data[col].dropna().unique()) <= {0, 1}]


data[binary_cols] = data[binary_cols]

v1 = data.loc[0, binary_cols].astype(int)
v2 = data.loc[1, binary_cols].astype(int)


f11 = np.sum((v1 == 1) & (v2 == 1))
f00 = np.sum((v1 == 0) & (v2 == 0))
f10 = np.sum((v1 == 1) & (v2 == 0))
f01 = np.sum((v1 == 0) & (v2 == 1))

jc = f11 / (f01 + f10 + f11)
smc = (f11 + f00) / (f00 + f01 + f10 + f11)

print("Jaccard Coefficient (JC):", jc)
print("Simple Matching Coefficient (SMC):", smc)
#Q9
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

data=pd.read_excel(r"Lab Session Data.xlsx", sheet_name='thyroid0387_UCI')
data.replace('?', np.nan, inplace=True)

label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    if data[column].isnull().sum() > 0:
        data[column].fillna(data[column].mode()[0], inplace=True)
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
    
    for column in data.select_dtypes(include=[np.number]).columns:
        data[column].fillna(data[column].mean(), inplace=True)
    
    v1 = data.iloc[0].values.reshape(1, -1)
v2 = data.iloc[1].values.reshape(1, -1)


cos_sim = cosine_similarity(v1, v2)[0][0]

print("Cosine Similarity:", cos_sim)
#Q10

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt


data=pd.read_excel(r"Lab Session Data.xlsx", sheet_name='thyroid0387_UCI')
data.replace('?', np.nan, inplace=True)

binary_cols = [col for col in data.columns if set(data[col].dropna().unique()) <= {'t', 'f'}]
binary_cols += [col for col in data.columns if set(data[col].dropna().unique()) <= {'M', 'F'}]

data[binary_cols] = data[binary_cols].replace({'t': 1, 'f': 0, 'M': 1, 'F': 0})

label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    if data[column].isnull().sum() > 0:
        data[column].fillna(data[column].mode()[0], inplace=True)
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

for column in data.select_dtypes(include=[np.number]).columns:
    data[column].fillna(data[column].mean(), inplace=True)

subset_data = data.iloc[:20]
jc_matrix = np.zeros((20, 20))
smc_matrix = np.zeros((20, 20))
cos_matrix = np.zeros((20, 20))

for i in range(20):
    for j in range(20):
        if i != j:
            v1 = subset_data.iloc[i, :].values
            v2 = subset_data.iloc[j, :].values

            v1_bin = subset_data.loc[i, binary_cols].astype(int)
            v2_bin = subset_data.loc[j, binary_cols].astype(int)

            f11 = np.sum((v1_bin == 1) & (v2_bin == 1))
            f00 = np.sum((v1_bin == 0) & (v2_bin == 0))
            f10 = np.sum((v1_bin == 1) & (v2_bin == 0))
            f01 = np.sum((v1_bin == 0) & (v2_bin == 1))

            jc_matrix[i, j] = f11 / (f01 + f10 + f11)
            smc_matrix[i, j] = (f11 + f00) / (f00 + f01 + f10 + f11)

            
            v1_all = v1.reshape(1, -1)
            v2_all = v2.reshape(1, -1)
            cos_matrix[i, j] = cosine_similarity(v1_all, v2_all)[0][0]

fig, axs = plt.subplots(1, 3, figsize=(20, 6))

sns.heatmap(jc_matrix, annot=True, cmap='viridis', ax=axs[0])
axs[0].set_title('Jaccard Coefficient (JC)')

sns.heatmap(smc_matrix, annot=True, cmap='viridis', ax=axs[1])
axs[1].set_title('Simple Matching Coefficient (SMC)')

sns.heatmap(cos_matrix, annot=True, cmap='viridis', ax=axs[2])
axs[2].set_title('Cosine Similarity (COS)')

plt.show()
