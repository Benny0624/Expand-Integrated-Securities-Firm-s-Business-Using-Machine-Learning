#!/usr/bin/env python
# coding: utf-8

# Import the libraries
import os
import numpy as np
import pandas as pd
import visuals as vs
import xlsxwriter
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from collections import Counter
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False



# Preprocessing
## Create group_by df
def Group_by(df, col):
    df_flat = df.groupby(['Customer_ID', col])['Buy_Amount'].sum().unstack()
    df_flat['Buy_Amount'] = df.groupby(['Customer_ID'])['Buy_Amount'].sum()
    df_flat['Age'] = df.groupby(['Customer_ID'])['Age'].first()
    df_flat.fillna(value=1, inplace=True)
    return df_flat

## Create scatter plot
def Scatter_plot(df, scaled = False):
    if scaled == True:
        # bc = PowerTransformer(method='box-cox')
        df_scaled = BC.fit_transform(df.values)
        df_scaled_df = pd.DataFrame(df_scaled, index=df.index, columns=df.columns)
        name = NAME_SPLT + '_' + 'scaled'
    else:
        df_scaled_df = df
        name = NAME_SPLT
    # Scatter_matrix plot of all features
    sm = pd.plotting.scatter_matrix(df_scaled_df, alpha=0.2, figsize=(8, 8), diagonal = 'kde')

    # Change label rotation
    [s.xaxis.label.set_rotation(45) for s in sm.reshape(-1)]
    [s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]

    # May need to offset label when rotating to prevent overlap of figure
    [s.get_yaxis().set_label_coords(-0.3,0.5) for s in sm.reshape(-1)]

    # Hide all ticks
    [s.set_xticks(()) for s in sm.reshape(-1)]
    [s.set_yticks(()) for s in sm.reshape(-1)]

    plt.savefig(name + '.png')
    return df_scaled_df

## Outlier detection
def Outlier_sheet(df, df_flat, df_scaled):
    scaledfeat_w_prop_df = df_flat.copy(deep = True)
    scaledfeat_w_prop_df['Customer_Name'] = df.groupby(['Customer_ID'])['Customer_Name'].first()
    scaledfeat_w_prop_df['Age'] = df.groupby(['Customer_ID'])['Age'].first()
    scaledfeat_w_prop_df['Gender'] = df.groupby(['Customer_ID'])['Gender'].first()
    scaledfeat_w_prop_df['Constellation'] = df.groupby(['Customer_ID'])['Constellation'].first()
    scaledfeat_w_prop_df['Buy_Amount'] = df.groupby(['Customer_ID'])['Buy_Amount'].sum()
    scaledfeat_w_prop_df['Com_ID'] = df.groupby(['Customer_ID'])['Com_ID'].first()
    scaledfeat_w_prop_df['Sales_ID'] = df.groupby(['Customer_ID'])['Com_ID'].first()
    scaledfeat_w_prop_df['Sales_Name'] = df.groupby(['Customer_ID'])['Com_ID'].first()

    feature_outliers = []
    writer = pd.ExcelWriter('Outliers_sheets.xlsx', engine='xlsxwriter')

    for feature in df_scaled.keys():
        Q1 = np.percentile(df_scaled[feature], 25)
        Q3 = np.percentile(df_scaled[feature], 75)
        step = 1.5*(Q3 - Q1)
        
        Outlier_Index = ~((df_scaled[feature] >= Q1 - step)&                           (df_scaled[feature] <= Q3 + step))
        scaledfeat_w_prop_df.loc[df_scaled.loc[Outlier_Index].index,:].groupby(['Customer_ID']).first()        .to_excel(writer, sheet_name = feature)
        feature_outliers.append(scaledfeat_w_prop_df.loc[df_scaled.loc[Outlier_Index].index,:])
    writer.save()

    # Flatten list of outliers
    outliers_flattened = []

    for i, j in enumerate(feature_outliers):
        outliers_flattened.append(feature_outliers[i].index)
    flat_list = [item for sublist in outliers_flattened for item in sublist]

    # Count the number of features for which a given observation is considered an outlier
    outlier_count = Counter(flat_list)
    outliers = [observation for observation in outlier_count.elements() if outlier_count[observation] >= 3]
    
    # Save the outliers sheet
    scaledfeat_w_prop_df.loc[df_scaled.loc[outliers].index,:].groupby(['Customer_ID']).first()    .to_csv('Outliers.csv', encoding='utf_8_sig')
    return scaledfeat_w_prop_df

## PCA procedure
def PCA_procedure(df_scaled, full = False):
    num_features_original = df_scaled.shape[1]
    num_features_two = 2
        
    pca_original = PCA(n_components = num_features_original, random_state = 0)
    pca_two = PCA(n_components = num_features_two, random_state = 0)
    
    if full == True:
        pca = pca_original.fit(df_scaled)
        vs.pca_results(df_scaled, pca)
        plt.savefig(NAME_SPLT+ '_PCA' + '.png')
    else:
        pca = pca_two.fit(df_scaled)
        df_scaled_copy = df_scaled.copy(deep = True)
        reduced_data = pca.transform(df_scaled_copy)
        reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
        vs.biplot(df_scaled ,reduced_data, pca)
        plt.savefig(NAME_SPLT+ '_PCA_Biplot' + '.png')
        return reduced_data
    return None

## Clustering
def Clustering(df_scaled, df_flat, reduced_data, elbow = False):
    if elbow == True:
        distortions = []
        K = range(2,100)
        for k in K:
            kmeanModel = KMeans(n_clusters=k).fit(df_scaled)
            distortions.append(sum(np.min(cdist(df_scaled, kmeanModel.cluster_centers_, 'euclidean'), axis=1))/df_scaled.shape[0])
        # Plot the elbow
        plt.clf()
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.savefig(NAME_SPLT + '_Elbow_Method' + '.png')
    else:
        num_clusters = np.arange(2,30)
        kmeans_results = {}
        for size in num_clusters:
            kmeans = KMeans(n_clusters = size).fit(df_scaled)
            preds = kmeans.predict(df_scaled)
            kmeans_results[size] = metrics.silhouette_score(df_scaled, preds)

        best_size = max(kmeans_results, key = kmeans_results.get)

        optimized_kmeans = KMeans(n_clusters = best_size, random_state = 0).fit(df_scaled)
        kmeans_preds = optimized_kmeans.predict(df_scaled)
        kmeans_centers = optimized_kmeans.cluster_centers_
        vs.cluster_results(reduced_data, kmeans_preds, kmeans_centers)
        plt.savefig(NAME_SPLT + 'Clustering_result' + '.png')
        
        # Inverse transform the box-cox centers
        # bc = PowerTransformer(method='box-cox').fit(df_flat.iloc[:,:FEAT_NUM])
        true_centers = BC.inverse_transform(kmeans_centers)
        true_centers[np.isnan(true_centers)] = 1

        # Save the true centers
        writer = pd.ExcelWriter('Data_Recovery_sheets.xlsx', engine='xlsxwriter')
        segments = ['Segment {}'.format(i) for i in range(0,len(kmeans_centers))]
        true_centers = pd.DataFrame(np.round(true_centers), columns = df_flat.iloc[:,:FEAT_NUM].keys())
        true_centers.index = segments
        true_centers.to_excel(writer, sheet_name = 'true_centers')
        np.round(df_flat.iloc[:,:FEAT_NUM].mean()).to_excel(writer, sheet_name = 'Population_mean')
        (true_centers - np.round(df_flat.iloc[:,:FEAT_NUM].mean())).to_excel(writer, sheet_name = 'true_centers_Minus_mean')
        
        # Save the label mean 
        df_flat['Label'] = optimized_kmeans.labels_
        round(df_flat.groupby('Label').mean()).to_excel(writer, sheet_name = 'Label_mean')
        writer.save()
    return None

PATH = 'C:/Users/User/Desktop/Data&Code'
NAME = input("Please enter data name:")
NAME_SPLT = NAME.split('.')[0]
GROUP_COL = input("Please enter group by col:")

## Change directory
os.chdir(PATH)
print("Current Working Directory " , os.getcwd())

## Import data 
DF = pd.read_excel(os.path.join(PATH, NAME))
# DF = DF.drop(DF.columns[0], axis=1)
DF["Tmp_Category"] = DF['Product_Category'] + "_" + DF["Product_sub_Category"]

## Group by
DF_FLAT = Group_by(DF, GROUP_COL)

## Scatter plot of features(w/wo box-cox transformation)
Scatter_plot(DF_FLAT)
BC = PowerTransformer(method='box-cox')
DF_SCALED = Scatter_plot(DF_FLAT,scaled = True)
FEAT_NUM = DF_SCALED.shape[1]

## Outliers detection
DF_WFEAT_SCALED = Outlier_sheet(DF, DF_FLAT, DF_SCALED)

## PCA procedure
PCA_procedure(DF_SCALED, full = True)
DF_REDUCED = PCA_procedure(DF_SCALED)

## Clustering
Clustering(DF_SCALED, DF_FLAT, DF_REDUCED, elbow = True)
Clustering(DF_SCALED, DF_FLAT, DF_REDUCED)