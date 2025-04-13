# The file includes functions for modeling and
# visualization of K-means time series clustering.

# Be sure to install any necessary packages first!
# Native libraries
import math
# Essential Libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
# Algorithms
from sklearn.metrics import silhouette_score
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans

# Global constants
SCHOOL_TERMS = ['K', 'K-S', '1', '1-S', '2', '2-S', '3', '3-S', '4', '4-S', '5']
TRAJGPS = ['Becoming Healthier', 'Early-onset OW/OB', 'Chronically OW/OB',
           'Consistently Healthy', 'Late-onset OW/OB']

# Calculate BIC.
def calculate_log_likelihood(X, n_clusters, centers, dataset):
  """ 
  Calculate the log likelihood of the clustering model

  Args:
      X (2D numpy array): observations of time-series target data
      n_clusters (int): the number of clusters
      centers (ndarray): cluster centers of the model
      dataset (dataframe): the entire dataset

  Returns:
      log_likelihood (float): the log likelihood of the clustering model
  """
  log_likelihood = 0.0
  
  # accumulate the log likelihood based on within-cluster SSR
  for i in range(n_clusters):
    indices = dataset[dataset['cluster'] == i].index
    cluster_points = X[indices]
    if len(cluster_points) > 0:
      deviation = cluster_points - centers[i].T
      log_likelihood += -0.5 * np.sum(deviation ** 2)

  return log_likelihood

def calculate_bic(X, n_clusters, centers, dataset):
  """
  Calculate the Bayesian Information Criterion of the clustering model

  Args:
      X (2D numpy array): observations of time-series target data
      n_clusters (int): the number of clusters
      centers (ndarray): cluster centers of the model
      dataset (dataframe): the entire dataset

  Returns:
      bic (float): the Bayesian Information Criterion
  """
  n_samples = X.shape[0]
  log_likelihood = calculate_log_likelihood(X, n_clusters, centers, dataset)

  # Calculate BIC
  bic = log_likelihood - 0.5 * n_clusters * np.log(n_samples)

  return bic


def choose_num_clusters(target_data, dataset, n_clusters = np.arange(2,11), 
                        seed=0):
  """
  Pick the optimal number of clusters for the K-means time series clustering
  model. Determination of the number of clusters is based on : 1) Bayesian
  Information Criterion (BIC): 2ΔBIC > 10 with lowest absolute value
  2) a trajectory group size of at least 5% of the sample.
  Note: need to convert data from dataframe format into numpy arrays
  using .values to fit into the function

  Args:
      target_data (2D numpy array): observations of time-series target data
      dataset (dataframe): the entire dataset
      n_clusters (numpy array, optional): a range of number of clusters for comparison.
      Defaults to np.arange(2,11).                       
      seed (int, optional): seed for replications. Defaults to 0.

  Returns:
      optimal_k (int): optimal number of clusters
  """
  # set the seed
  np.random.seed(seed)
  
  # initialize variables for tracking 
  optimal_k = 0
  bics = []
  
  for i in n_clusters:
    # track the existence of minority class
    has_minor_cluster = False
    
    # apply k-means time series to generate clusters 
    model = TimeSeriesKMeans(n_clusters=i, random_state=seed)
    model.fit(target_data)
    
    # get cluster centers and map labels to the original data
    labels = model.fit_predict(target_data)
    centers = model.cluster_centers_
    dataset['cluster'] = labels
    
    # evaluate cluster size 
    for num in range(i):
      cluster_percent = len(dataset[dataset['cluster'] == num]) / len(dataset)
      if cluster_percent < 0.05:
        has_minor_cluster = True
    
    # calculate BIC
    bic = calculate_bic(target_data, i, centers, dataset)
    bics.append(bic)
    
    # check if there is any minority cluster
    if not has_minor_cluster:
      if i == n_clusters[0]:
        optimal_k = i
      else:
        # evaluate BIC and ΔBIC
        if abs(bic - bics[-2]) > 10 and abs(bic) < abs(bics[-2]):
          optimal_k = i
  for i in range(2,7):
    print(f'BIC with {i} clusters: {bics[i-2]}')
  return optimal_k


def plot_kmeans_ts_clustering(data, target, num_clusters, 
                              dist_metric='euclidean', seed=0, cluster_names=[]):
  """
  Create the clusters give the parameters and plot the clusters.

  Args:
      data (a list of lists): target data
      target (str): name of the target
      num_clusters (int): number of clusters
      Defaults to NUM_CLUSTERS.  
      dist_metric (str, optional): distance metric. Defaults to 'euclidean'.
      seed (int, optional): seed for replications. Defaults to 0.
      cluster_names (list, optional): cluster names. Defaults to [].

  Returns:
      labels (ndarray): Index of the cluster each sample belongs to
  """
  # set the seed
  np.random.seed(seed)
  
  # build the model
  km = TimeSeriesKMeans(n_clusters=num_clusters, metric=dist_metric, random_state=seed)
  labels = km.fit_predict(data)
  plot_count = math.ceil(math.sqrt(num_clusters))
  
  # Plot the results
  fig, axs = plt.subplots(plot_count,plot_count,figsize=(25,25))
  title = 'Clusters of ' + target
  fig.suptitle(title, fontsize=30)
  row_i=0
  column_j=0
  
  for label in set(labels):
    cluster = []
    for i in range(len(labels)):
      
      # map individual data to cluster labels
      if(labels[i]==label):
        axs[row_i, column_j].plot(data[i],c="gray",alpha=0.4)
        cluster.append(data[i])
        
    if len(cluster) > 0:
      # DTW tracks cluster centers with DBA 
      if dist_metric == 'dtw':
        axs[row_i, column_j].plot(dtw_barycenter_averaging(np.vstack(cluster)),c="red")
      
      else: # Euclidean distance takes arithmetic mean for cluster centers
        axs[row_i, column_j].plot(np.average(np.vstack(cluster),axis=0),c="red")
        
    if cluster_names == []:
      axs[row_i, column_j].set_title("Cluster "+str(row_i*plot_count+column_j), fontsize=20)
    else:
      axs[row_i, column_j].set_title(cluster_names[row_i*plot_count+column_j], fontsize=20)
    
    # customize y-ticks
    if math.ceil(np.max(data)) > 20: # y-axis are percentage values
      y_ticks = 10 * np.arange(math.floor(np.min(data)/10), math.ceil(np.max(data)/10+1), step=1) # Adjust as needed
    else:
      y_ticks = np.arange(math.floor(np.min(data)), math.ceil(np.max(data)+1), step=1)
    
    # set ticks
    axs[row_i, column_j].set_yticks(y_ticks)
    axs[row_i, column_j].set_xticks(np.arange(11), SCHOOL_TERMS, fontsize=16)
    column_j+=1
    
    # start a new row 
    if column_j%plot_count == 0:
        row_i+=1
        column_j=0
  plt.savefig("kmeans_trajgps.pdf", format="pdf")
  plt.show()
  return labels


def plot_prob_overweight_for_clusters(num_clusters, labels, dataset, cluster_names=[]):
  """
  Plot the probablity of overweight/obese over time across clusters.
  Args:
      num_clusters (int): number of clusters created
      labels (ndarray): Index of the cluster each sample belongs to
      dataset (dataframe): the entire dataset
      cluster_names (list, optional): cluster names. Defaults to [].
  """
  
  # update clusters in dataset
  dataset['cluster'] = labels
  prob_overweight = []
  
  # filter the indicator data of whether ow/ob
  bmipbool = bmipbool_time_series(dataset)

  # calculate probability of overweight/obese over time for each group
  for i in range(num_clusters):
    # cluster size
    cluster_size = len(dataset[dataset['cluster'] == i])
    
    # store the probability of ow/ob for each time point in the current cluster
    prob_overweight_cluster_i = []
    
    for j in range(bmipbool.shape[1]):
      # track ow/ob cases
      count_overweight = 0
      for index, row in dataset.iterrows():
        if row['cluster'] == i and bmipbool.iloc[index, j] == 1:
          count_overweight += 1
          
      prob_overweight_cluster_i.append(count_overweight/cluster_size)
    prob_overweight.append(prob_overweight_cluster_i)
  
  # plot the results
  legends = []
  for i in range(num_clusters):
    if cluster_names == []:
      legends.append('Cluster ' + str(i))
    else:
      legends.append(cluster_names[i])
  fig, ax = plt.subplots()
  
  for i in range(len(prob_overweight)):
    ax.plot(prob_overweight[i], label=legends[i])
    
  y_ticks = np.arange(0, 120, step=20)/100  # Adjust as needed
  
  # set ticks
  ax.set_yticks(y_ticks)
  ax.set_xticks(np.arange(11), SCHOOL_TERMS)
  
  # adjust plot attributes
  ax.set_xlabel("School Terms")
  ax.set_ylabel("Probability of Overweight/Obese")
  #ax.set_title("Probability of Overweight/Obese in Different Clusters Over Time")
  plt.legend()
  plt.savefig("ow_prob_trajgps.pdf", format="pdf")
  plt.show()


def compare_dist_metric(num_clusters, data, dist_metric = ['euclidean', 'dtw'], seed=0):
  """
  Use Silhouette Score to compare differernt distance metrics.
  Silhouette Score measures how well-separated the clusters are.
  The value ranges from -1 to 1, with a higher silhouette score
  indicating better-defined clusters.

  Args:
      num_clusters (int): number of clusters
      data (dataframe): observations of target data
      dist_metric (list, optional): Distance metrics to compare. 
      Defaults to ['euclidean', 'dtw'].
      seed (int, optional): For replications. 
      Defaults to 0.
  """
  # set the seed
  np.random.seed(seed)
  
  silhouette_scores = []
  
  # get silhouette scores
  for i in dist_metric:
    # fit the model
    km= TimeSeriesKMeans(n_clusters=num_clusters, metric=i, random_state=seed)
    labels = km.fit_predict(data)
    silhouette = silhouette_score(data, labels)
    silhouette_scores.append(silhouette)
    
  # plot the results
  fig, ax = plt.subplots()
  ax.bar(dist_metric, silhouette_scores)
  
  # set plot attributes
  y_ticks = np.arange(math.ceil(np.max(silhouette_scores)*10+1), step=1)/10 # Adjust as needed
  ax.set_yticks(y_ticks)
  ax.set_ylabel("Silhouette Score")
  ax.set_title("Silhouette Score with Different Distance Metrics")
  
  plt.show()


def dataframe_to_list(data_frame):
  """
  Given a 2D dataframe, convert it to a list of the contents of its rows

  Args:
    data_frame (dataframe): a dataset
  
  Returns:
    df_to_list (list of lists): a list where every inner list is a row entry
  """
  df_to_list = []
  for i in range(len(data_frame)):
    df_to_list.append(data_frame.iloc[i].values)
  return df_to_list


def bmiz_time_series(student_data):
  """
  Gets a subset of the data for just bmiz over time

  Args:
    student_data (dataset): dataset of students

  Returns: 
    a subset of student_data of just columns measuring bmiz over time

  """
  pattern_bmiz = r'^bmiz_[0-9]+$'
  return student_data.filter(regex=pattern_bmiz)


def bmipbool_time_series(student_data):
  """
  Gets a subset of the data for just bmipbool over time

  Args:
    student_data (dataset): dataset of students

  Returns: 
    a subset of student_data of just columns measuring bmipbool over time
  """
  pattern_bmipbool = r'^bmipbool_[0-9]+$'
  return student_data.filter(regex=pattern_bmipbool)


def add_changes_columns(student_data):
  """
  Adds columns of bmiz changes, bmi changes, and bmi percentage changes to the student dataset

  Args:
    student_data (dataset): dataset of students
  """
  # get BMIz time series
  bmiz = bmiz_time_series(student_data)

  # add columns of bmiz changes, bmi changes, and bmi percentage changes
  for i in range(1, bmiz.shape[1]):
    student_data['chg_bmiz_' + str(i)] = bmiz.loc[:, 'bmiz_' + str(i+1)] - bmiz.loc[:, 'bmiz_' + str(i)]
  for j in range(1, bmiz.shape[1]):
    student_data['percent_chg_bmi_' + str(j)] = 100*(student_data.loc[:, 'BMI_' + str(j+1)] - 
                student_data.loc[:, 'BMI_' + str(j)])/student_data.loc[:, 'BMI_' + str(j)] 
  for k in range(1, bmiz.shape[1]):
    student_data['chg_bmi_' + str(k)] = student_data.loc[:, 'BMI_' + str(k+1)] - student_data.loc[:, 'BMI_' + str(k)]
  
  
def compute_silhouette(data, n_clusters, dist_metric='euclidean', init='k-means++', seed=0):
  """
  Compute the silhouette score of the K-means time series model 
  given the data and the hyperparameters.
  
  Args:
      data (dataframe): observations of target data
      n_clusters (int): the number of clusters for the model
      dist_metric (str, optional): _description_. Defaults to 'euclidean'.
      init (str, optional): method for initialization of centroids. Defaults to 'k-means++'.
      seed (int, optional): generator used to initialize the centers, 
      also for result replications. Defaults to 0.

  Returns:
      silhouette_score (float): the silhouette score of the model
  """
  # fit the model
  km = TimeSeriesKMeans(n_clusters=n_clusters, metric=dist_metric, init=init, random_state = seed)
  labels = km.fit_predict(data)
  silhouette = silhouette_score(data, labels)
  
  return silhouette


def tune_dist_metric(data, n_clusters, dist_metric=['euclidean', 'dtw'], init='k-means++', seed=np.arange(100)):
  """
  Evaluate the performance of the model across different distance metrics using silhouette scores. 
  
  Args:
      data (dataframe): observations of target data
      n_clusters (int): the number of clusters for the model
      dist_metric (list, optional): Distance metrics to compare. 
      Defaults to ['euclidean', 'dtw'].
      init (str, optional): method for initialization of centroids. Defaults to 'k-means++'.
      seed (int array, optional): an array of integers as generators to initialize the centers, 
      also for result replications.. Defaults to np.arange(100).

  Returns:
      silhouettes (dataframe): silhouette scores for different metrics
  """
  # store model evaluation measurements
  silhouettes = {}
  
  for i in dist_metric:
    silhouettes[i] = []
    for j in seed:
      # compute the silhouette score
      silhouette = compute_silhouette(data, n_clusters, i, init, j)
      
      # update the dictionary
      silhouettes[i].append(silhouette)
  
  # make a dataframe
  silhouettes_df= pd.DataFrame(silhouettes)
  
  return silhouettes_df


def plot_dist_metric_tunning(dist_metric_evals):
  """
  Plot the evaluation metric measurements across different distance metrics
  to visualize which distance metric performs the best. 
   
  Args:
      dist_metric_evals (dataframe): evaluation metric measurements for comparison
  """
  
  x_labels = dist_metric_evals.columns
  
  # Calculating means and standard deviations
  silhouette_euclidean = dist_metric_evals[x_labels[0]]
  silhouette_dtw = dist_metric_evals[x_labels[1]]
  mean_silhouette_euclidean, mean_silhouette_dtw = np.mean(silhouette_euclidean), np.mean(silhouette_dtw)
  std_silhouette_euclidean, std_silhouette_dtw = np.std(silhouette_euclidean), np.std(silhouette_dtw)
  n = len(silhouette_dtw)

  # Calculating 95% confidence intervals
  ci_silhouette_euclidean = 1.96 * (std_silhouette_euclidean / np.sqrt(n))  
  ci_silhouette_dtw = 1.96 * (std_silhouette_dtw / np.sqrt(n))

  # Plotting
  fig = plt.figure(figsize=(15, 8))
  plt.errorbar([1, 2], [mean_silhouette_euclidean, mean_silhouette_dtw ], 
               yerr=[ci_silhouette_euclidean, ci_silhouette_dtw], fmt='o', ms=15, capsize=15)
  plt.xticks([1, 2], ['Euclidean', 'DTW'])
  plt.ylabel('Silhouette Score', fontsize=36)
  plt.xlim(0, 3)
  
  y_ticks = [0.62, 0.63, 0.64, 0.65, 0.66, 0.67]
  plt.yticks(y_ticks)
  plt.tick_params(axis='x', which='both', labelsize=28)
  plt.tick_params(axis='y', which='both', labelsize=28)
  
  plt.savefig("kmeans_dist_metric.pdf", format="pdf")
  plt.show()
  print(f"The average silhouette score using euclidean distance is {mean_silhouette_euclidean:.3f}"
        f", with a 95% confidence interval [{mean_silhouette_euclidean-ci_silhouette_euclidean:.3f}, "
        f"{mean_silhouette_euclidean+ci_silhouette_euclidean:.3f}].")
  
  print(f"The average silhouette score using DTW is {mean_silhouette_dtw:.3f}"
        f", with a 95% confidence interval [{mean_silhouette_dtw-ci_silhouette_dtw:.3f}, "
        f"{mean_silhouette_dtw+ci_silhouette_dtw:.3f}].")

    
def tune_initialization(data, n_clusters, init=['k-means++','random'], dist_metric='euclidean', seed=np.arange(100)):
  """
  Evaluate the performance of the model across a variety of initial state of cluster centers.
  Initializations differ by the seeds served as random generators. Evaluation metrics include
  silhouette scores, inertias, and the number of iterations to converge.
  
  Args:
      data (dataframe): observations of target data
      n_clusters (int): the number of clusters for the model
      dist_metric (str, optional): distance metric used in the model. Defaults to 'euclidean'.
      init (list, optional): method for initialization of centroids. Defaults to ['k-means++', 'random'].
      seed (int array, optional): an array of integers as generators to initialize the centers, 
      also for result replications.. Defaults to np.arange(100).

  Returns:
      silhouettes_df(dataframe): silhouette metric measurements for comparison
  """
  # store model evaluation measurements
  silhouettes = {}
  
  seed_to_drop = []
  for i in init:
    silhouettes[i] = []
    for j in seed:
      
      km = TimeSeriesKMeans(n_clusters, metric=dist_metric, init=i, random_state=j)
      labels = km.fit_predict(data)
      
      if len(np.unique(labels)) < 2:
        seed_to_drop.append(j)
        continue
      # compute the silhouette score
      silhouette = compute_silhouette(data, n_clusters, dist_metric, i, j)
      
      # update the dictionary
      silhouettes[i].append(silhouette)
  
  invalid_seed = list(set(seed_to_drop))
  
  silhouettes['k-means++'] = [value for i, value in enumerate(silhouettes['k-means++']) if i not in invalid_seed]
  
  # make a dataframe
  silhouettes_df= pd.DataFrame(silhouettes)
  
  return silhouettes_df


def plot_initialization_tunning(evals):
  """
  Plot the evaluation metric measurements across different initialization states
  to visualize which seed performs the best.
  
  Args:
      evals (dataframe): evaluation metric measurements for comparison
  """
  
  x_labels = evals.columns
  
  # Calculating means and standard deviations
  silhouette_kmeans_pp = evals[x_labels[0]]
  silhouette_random = evals[x_labels[1]]
  mean_silhouette_kmeans_pp, mean_silhouette_random = np.mean(silhouette_kmeans_pp), np.mean(silhouette_random)
  std_silhouette_kmeans_pp, std_silhouette_random = np.std(silhouette_kmeans_pp), np.std(silhouette_random)
  n = len(silhouette_kmeans_pp)

  # Calculating 95% confidence intervals
  ci_silhouette_kmeans_pp = 1.96 * (std_silhouette_kmeans_pp / np.sqrt(n))  
  ci_silhouette_random = 1.96 * (std_silhouette_random / np.sqrt(n))

  # Plotting
  fig = plt.figure(figsize=(15, 8))
  plt.errorbar([1, 2], [mean_silhouette_kmeans_pp, mean_silhouette_random], 
               yerr=[ci_silhouette_kmeans_pp, ci_silhouette_random], fmt='o', ms=15, capsize=15)
  plt.xticks([1, 2], ['k-means++', 'random k\n observations from data'])
  plt.ylabel('Silhouette Score', fontsize=36)
  plt.xlim(0, 3)
  
  y_ticks = [0.62, 0.63, 0.64, 0.65, 0.66, 0.67]
  plt.yticks(y_ticks)
  
  plt.tick_params(axis='x', which='both', labelsize=28)
  plt.tick_params(axis='y', which='both', labelsize=28)
  
  plt.savefig("kmeans_initialization.pdf", format="pdf")
  plt.show()
  print(f"The average silhouette score using k-means++ is {mean_silhouette_kmeans_pp:.3f}"
        f", with a 95% confidence interval [{mean_silhouette_kmeans_pp-ci_silhouette_kmeans_pp:.3f}, "
        f"{mean_silhouette_kmeans_pp+ci_silhouette_kmeans_pp:.3f}].")
  
  print(f"The average silhouette score using random k observations from data is {mean_silhouette_random:.3f}"
        f", with a 95% confidence interval [{mean_silhouette_random-ci_silhouette_random:.3f}, "
        f"{mean_silhouette_random+ci_silhouette_random:.3f}].")


def map_cluster_group(dataset, value_mapping = {0: 1, 1: 4, 2: 5, 3: 2, 4: 3}):
  """
  Map the labels between k-means clusters and the existing GBTM groups 
  to maintain consistency for comparison.

  Args:
      dataset (dataframe): the entire dataset
      value_mapping (dict, optional): A mapping from cluster indices to 
      group indices. Defaults to {0: 1, 1: 4, 2: 5, 3: 2, 4: 3}.
  """
  # Use map to modify the column
  dataset['cluster'] = dataset['cluster'].map(value_mapping).astype(int)
  
  
def check_diff(dataset):
  """
  Check if the labels are different between the two models and 
  add a column of booleans to indicate the difference.

  Args:
      dataset (dataframe): the entire dataset
  """
  dataset['cluster_diff'] = np.where(dataset['cluster'] != dataset['GROUP'], 1, 0)


def filter_diff(dataset):
  """
  Filter the subset of data labeled differently between the models.

  Args:
      dataset (dataframe): the entire dataset
  Returns:
      dataframe: subset of data labeled differently between the models
  """
  return dataset[dataset['cluster_diff'] == 1]


def pivot_diff(dataset):
  """
  Focus on the students differed in labeling and 
  breakdown the difference among group categories.

  Args:
      dataset (dataframe): the entire dataset
      
  Returns:
      pivot_table (dataframe): a pivot table that details the differences in labeling
  """
  
  # filter students who have different labels between the models
  students_differ = filter_diff(dataset)
  
  # specify which group each index corresponds to
  group_names = ['Becoming Healthier', 'Consistently Healthy', 
                 'Late-onset OW/OB', 'Early-onset OW/OB',
                 'Chronically OW/OB']
  
  # create a pivot table to visualize the differences
  pivot_table = pd.pivot_table(students_differ[['cluster','GROUP']], 
                               index='cluster', columns='GROUP', aggfunc=len, 
                               fill_value=0, margins=True, margins_name='Total')
  
  # change row and column names
  col_names = [group_names[col-1] for col in pivot_table.columns 
               if type(col)==int] + ['Total']
  row_names = [group_names[index-1] for index in pivot_table.index 
               if type(index)==int] + ['Total']
  pivot_table = pivot_table.rename(index=dict(zip(pivot_table.index, row_names)), 
                                   columns=dict(zip(pivot_table.columns, col_names)))
  pivot_table.columns.name = "GBTM"
  pivot_table.index.name = "K-means"
  
  return pivot_table


def calc_diff_percent(dataset):
  """
  Calculate the percentage of observations differed in labeling.

  Args:
      dataset (dataframe): the entire dataset

  Returns:
      float: percentage of observations differed in labeling.
  """
  return dataset['cluster_diff'].sum() / dataset['cluster_diff'].count()


def silhouette_existing_labels(data):
  """
  Calculates the silhouette score for the existing labels

  Args:
      data (dataframe): the entire dataset
  
  Returns:
      float: the silhouette score

  """
  bmipbool = bmipbool_time_series(data)
  labels = data['moreno_traj_group']
  return silhouette_score(bmipbool, labels)


def map_group_cluster(dataset, value_mapping = {1: 0, 2: 3, 3: 4, 4: 1, 5: 2}):
    """
    Map GBTM membership labels to k-means membership labels
    to maintain consistency for comparison.
    
    Args:
        dataset (dataframe): the entire dataset
        value_mapping (dict, optional): A mapping from GBTM indices to 
        k-means indices. Defaults to {1: 0, 2: 3, 3: 4, 4: 1, 5: 2}.
    """
    # Use map to modify the column
    dataset['GROUP'] = dataset['GROUP'].map(value_mapping).astype(int)
    
    
def cluster_membership_percent_gbtm(data):
  """
  Calculate GBTM group membership percentages
      
  Args:
      data (dataframe): the entire dataset

  Returns:
      dictionary: GBTM group membership percentages
  """
  cluster_membership_gbtm = {}
  for i in range(len(TRAJGPS)):
    cluster_size = len(data[data['GROUP'] == i])
    percent = cluster_size/len(data) * 100
    cluster_membership_gbtm[TRAJGPS[i]] = percent
  return cluster_membership_gbtm
    
    
def cluster_membership_percent_kmeans(data):
  """
  Calculate k-means group membership percentages
      
  Args:
      data (dataframe): the entire dataset

  Returns:
      dictionary: k-means group membership percentages
  """
  cluster_membership_kmeans = {}
  for i in range(len(TRAJGPS)):
    cluster_size = len(data[data['cluster'] == i])
    percent = cluster_size/len(data) * 100
    cluster_membership_kmeans[TRAJGPS[i]] = percent
  return cluster_membership_kmeans


def kmeans_bmiz_centers(data):
  """
  Map k-means cluster membership to bmiz and calculate the centers

  Args:
      data (dataframe): the entire dataset

  Returns:
      list: k-means centroids of trajectory groups 
  """
    
  kmeans_bmiz_centroids = []
  for i in range(len(TRAJGPS)):
    # cluster size
    students_in_i = data[data['cluster'] == i]
    bmiz_in_i = bmiz_time_series(students_in_i)
    cluster_size = len(students_in_i)
    
    cluster_mean = np.array(bmiz_in_i.sum())/cluster_size
    kmeans_bmiz_centroids.append(cluster_mean)
  return kmeans_bmiz_centroids


def gbtm_bmiz_centers(data):
  """
  Map gbtm group membership to bmiz and calculate the centers

  Args:
      data (dataframe): the entire dataset

  Returns:
      list: gbtm centroids of trajectory groups 
  """
    
  gbtm_bmiz_centroids = []
  for i in range(len(TRAJGPS)):
    # group size
    students_in_i = data[data['GROUP'] == i]
    bmiz_in_i = bmiz_time_series(students_in_i)
    group_size = len(students_in_i)
    
    group_mean = np.array(bmiz_in_i.sum())/group_size
    gbtm_bmiz_centroids.append(group_mean)
  return gbtm_bmiz_centroids


def dist_from_center_by_cluster(data):
  """
  Calculate k-means and GBTM distances from centroids by cluster
  
  Args:
      data (dataframe): the entire dataset

  Returns:
      tuple: a tuple of k-means and GBTM lists of distance from centroids by cluster
  """
  kmeans_dist_from_center_by_cluster = []
  gbtm_dist_from_center_by_cluster = []
  bmiz_to_list = dataframe_to_list(bmiz_time_series(data))
    
  for label in set(range(5)):
    dist_kmeans_label = []
    dist_gbtm_label = []

    for i in range(len(data)):
      # map individual data to k-means labels
      if(data['cluster'][i] == label):
        dist = np.linalg.norm(bmiz_to_list[i] - kmeans_bmiz_centers(data)[label])
        dist_kmeans_label.append(dist)
      # map individual data to gbtm labels
      if(data['GROUP'][i] == label):
        dist = np.linalg.norm(bmiz_to_list[i] - gbtm_bmiz_centers(data)[label])
        dist_gbtm_label.append(dist)
                
    kmeans_dist_from_center_by_cluster.append(dist_kmeans_label)
    gbtm_dist_from_center_by_cluster.append(dist_gbtm_label)
  return (kmeans_dist_from_center_by_cluster, gbtm_dist_from_center_by_cluster)
  

def kmeans_mean_dist_by_cluster(kmeans_dist_from_center_by_cluster):
  """
  Calculate k-means mean deviation (Euclidean distance) from centers for each cluster

  Args:
      kmeans_dist_from_center_by_cluster (list): list of observations' distance from
      k-means centroids by cluster 

  Returns:
      numpy array: k-means mean deviation (Euclidean distance) from centroids for each cluster
  """
  kmeans_mean_dist_from_center_per_cluster = np.array([np.mean(i) for i in kmeans_dist_from_center_by_cluster])
  return kmeans_mean_dist_from_center_per_cluster


def kmeans_cluster_sizes(kmeans_dist_from_center_by_cluster):
  """
  Calculate k-means cluster sizes
  
  Args:
      kmeans_dist_from_center_by_cluster (list): list of observations' distance from
      k-means centroids by cluster 

  Returns:
      numpy array: k-means cluster sizes
  """
  kmeans_cluster_size = np.array([len(i) for i in kmeans_dist_from_center_by_cluster])
  return kmeans_cluster_size


def kmeans_std_by_cluster(kmeans_dist_from_center_by_cluster, kmeans_mean_dist_from_center_per_cluster, kmeans_cluster_size):
  """
  Calculate k-means standard deviation per cluster
  
  Args:
      kmeans_dist_from_center_by_cluster (list): list of observations' distance from
      k-means centroids by cluster 
      kmeans_mean_dist_from_center_per_cluster (numpy array): k-means mean deviation 
      (Euclidean distance) from centroids for each cluster
      kmeans_cluster_size (numpy array): k-means cluster sizes

  Returns:
      numpy array: k-means standard deviation per cluster
  """
  # k-means sum of squared residuals per cluster
  kmeans_ssr_per_cluster = [np.sum((np.array(kmeans_dist_from_center_by_cluster[i]) 
                                    - kmeans_mean_dist_from_center_per_cluster[i])**2) 
                            for i in range(5)]
  
  kmeans_std_dev_per_cluster = np.sqrt(kmeans_ssr_per_cluster / kmeans_cluster_size)
  return kmeans_std_dev_per_cluster


def kmeans_accuracy(kmeans_dist_from_center_by_cluster, kmeans_mean_dist_from_center_per_cluster, 
                    kmeans_std_dev_per_cluster, kmeans_cluster_size, data):
  """
  Calculate k-means overall accuracy and accuracy by cluster 

  Args:
      kmeans_dist_from_center_by_cluster (list): list of observations' distance from
      k-means centroids by cluster 
      kmeans_mean_dist_from_center_per_cluster (numpy array): k-means mean deviation 
      (Euclidean distance) from centroids for each cluster
      kmeans_std_dev_per_cluster (numpy array): k-means standard deviation per cluster
      kmeans_cluster_size (numpy array): k-means cluster sizes
      data (dataframe): the entire dataset
      
  Returns:
      tuple: a tuple consists of an array of k-means accuracy by cluster and the overall accuracy
  """
  kmeans_accuracy_per_obs = [(np.array(kmeans_dist_from_center_by_cluster[i]) - kmeans_mean_dist_from_center_per_cluster[i]) 
                             <= 2*kmeans_std_dev_per_cluster[i] for i in range(5)]
  kmeans_accuracy_per_cluster = np.array([sum(kmeans_accuracy_per_obs[i]) for i in range(5)])/kmeans_cluster_size
  kmeans_overall_accuracy = sum([sum(kmeans_accuracy_per_obs[i]) for i in range(5)]) / len(data)
  
  return (kmeans_accuracy_per_cluster, kmeans_overall_accuracy)


def gbtm_mean_dist_by_cluster(gbtm_dist_from_center_by_cluster):
  """
  Calculate GBTM mean deviation (Euclidean distance) from centers for each cluster

  Args:
      gbtm_dist_from_center_by_cluster (list): list of observations' distance from
      GBTM centroids by cluster 

  Returns:
      numpy array: GBTM mean deviation (Euclidean distance) from centroids for each cluster
  """
  gbtm_mean_dist_from_center_per_cluster = np.array([np.mean(i) for i in gbtm_dist_from_center_by_cluster])
  return gbtm_mean_dist_from_center_per_cluster


def gbtm_cluster_sizes(gbtm_dist_from_center_by_cluster):
  """
  Calculate GBTM cluster sizes
  
  Args:
      gbtm_dist_from_center_by_cluster (list): list of observations' distance from
      GBTM centroids by cluster 

  Returns:
      numpy array: GBTM cluster sizes
  """
  gbtm_cluster_size = np.array([len(i) for i in gbtm_dist_from_center_by_cluster])
  return gbtm_cluster_size


def gbtm_std_by_cluster(gbtm_dist_from_center_by_cluster, 
                        gbtm_mean_dist_from_center_per_cluster, 
                        gbtm_cluster_size):
  """
  Calculate GBTM standard deviation per cluster
  
  Args:
      gbtm_dist_from_center_by_cluster (list): list of observations' distance from
      GBTM centroids by cluster 
      gbtm_mean_dist_from_center_per_cluster (numpy array): GBTM mean deviation 
      (Euclidean distance) from centroids for each cluster
      gbtm_cluster_size (numpy array): GBTM cluster sizes

  Returns:
      numpy array: gbtm standard deviation per cluster
  """
  # GBTM sum of squared residuals per cluster
  gbtm_ssr_per_cluster = [np.sum((np.array(gbtm_dist_from_center_by_cluster[i]) 
                                  - gbtm_mean_dist_from_center_per_cluster[i])**2) 
                          for i in range(5)]
  
  gbtm_std_dev_per_cluster = np.sqrt(gbtm_ssr_per_cluster / gbtm_cluster_size)
  return gbtm_std_dev_per_cluster


def gbtm_accuracy(gbtm_dist_from_center_by_cluster, 
                  gbtm_mean_dist_from_center_per_cluster, 
                  gbtm_std_dev_per_cluster, 
                  gbtm_cluster_size, data):
  """
  Calculate GBTM overall accuracy and accuracy by cluster 

  Args:
      gbtm_dist_from_center_by_cluster (list): list of observations' distance from
      GBTM centroids by cluster 
      gbtm_mean_dist_from_center_per_cluster (numpy array): GBTM mean deviation 
      (Euclidean distance) from centroids for each cluster
      gbtm_std_dev_per_cluster (numpy array): GBTM standard deviation per cluster
      gbtm_cluster_size (numpy array): GBTM cluster sizes
      data (dataframe): the entire dataset
      
  Returns:
      tuple: a tuple consists of an array of GBTM accuracy by cluster and the overall accuracy
  """
  gbtm_accuracy_per_obs = [(np.array(gbtm_dist_from_center_by_cluster[i]) - gbtm_mean_dist_from_center_per_cluster[i]) 
                           <= 2*gbtm_std_dev_per_cluster[i] for i in range(5)]
  gbtm_accuracy_per_cluster = np.array([sum(gbtm_accuracy_per_obs[i]) for i in range(5)])/gbtm_cluster_size
  gbtm_overall_accuracy = sum([sum(gbtm_accuracy_per_obs[i]) for i in range(5)]) / len(data)
  
  return (gbtm_accuracy_per_cluster, gbtm_overall_accuracy)


def accuracy_table(kmeans_accuracy, gbtm_accuracy):
  """
  Create an accuracy table

  Args:
      kmeans_accuracy (tuple): consists of an array of k-means accuracy by cluster and the overall accuracy
      gbtm_accuracy (tuple): consists of an array of GBTM accuracy by cluster and the overall accuracy

  Returns:
      _type_: _description_
  """
  accuracy = {
    'Method': ['K-Means', 'GBTM'],
    'Overall Accuracy (%)': [kmeans_accuracy[1], gbtm_accuracy[1]]
    }
  
  for i in range(5):
    col_name = TRAJGPS[i] + ' Accuracy (%)'
    accuracy[col_name] = [kmeans_accuracy[0][i], gbtm_accuracy[0][i]]
  
  # Create a DataFrame
  df = pd.DataFrame(accuracy)
  
  # Set 'Method' as the index
  df.set_index('Method', inplace=True)
  
  return df


def store_trajgps(data):
  """
  Store k-means and GBTM trajectory groups 

  Args:
      data (dataframe): the entire dataset 

  Returns:
      tuple : consists of k-means and GBTM trajectory groups
  """
  kmeans_trajs = []
  gbtm_trajs = []
  bmipbool = (bmipbool_time_series(data))
  
  for i in range(5):
    # cluster size
    cluster_size = len(data[data['cluster'] == i])
    group_size = len(data[data['GROUP'] == i])
    
    # store the probability of ow/ob for each time point in the current cluster/group
    prob_overweight_cluster_i = []
    prob_overweight_group_i = []
    
    for j in range(bmipbool.shape[1]):
      # track ow/ob cases
      count_overweight_cluster = 0
      count_overweight_group = 0
      for index, row in data.iterrows():
        if row['cluster'] == i and bmipbool.iloc[index, j] == 1:
          count_overweight_cluster += 1
        if row['GROUP'] == i and bmipbool.iloc[index, j] == 1:
          count_overweight_group += 1
          
      prob_overweight_cluster_i.append(count_overweight_cluster/cluster_size)
      prob_overweight_group_i.append(count_overweight_group/group_size)
    kmeans_trajs.append(prob_overweight_cluster_i)
    gbtm_trajs.append(prob_overweight_group_i)
  
  return (kmeans_trajs, gbtm_trajs)


def plot_refined_trajgps(trajgps):
  kmeans_data = trajgps[0]
  gbtm_data = trajgps[1]
  
  grades = ['Kindergarten', '1st', '2nd', '3rd', '4th', '5th']
  semesters = ['Fall', 'Spring', 'Fall', 'Spring', 'Fall', 'Spring', 'Fall', 'Spring', 'Fall', 'Spring', 'Fall']
  
  # Define trajectory groups and corresponding colors
  trajectory_groups = TRAJGPS
  trajectory_colors = ['blue', 'red', 'purple', 'green', 'orange']
  
  fig, ax = plt.subplots(figsize=(23,12))
  # Plot trajectories without labels
  for i, group in enumerate(trajectory_groups):
    # Plot K-Means with solid lines
    ax.plot(kmeans_data[i], color=trajectory_colors[i], linestyle='-', marker='o', lw=3, ms=15)

    # Plot GBTM with dashed lines
    ax.plot(gbtm_data[i], color=trajectory_colors[i], linestyle='--', marker='s', lw=3, ms=15)

  # Add labels and the custom legend (colors only)
  ax.set_ylabel("Probability of Overweight/Obese", fontsize=38)
  #plt.suptitle('Trajectory Groups of K-means vs. GBTM', fontsize=50)

  # Add spans for each fall-spring pair
  for i in range(len(grades)):
    if i < 5:
      ax.axvspan(i * 2 + 1, i * 2 + 2, facecolor='grey', alpha=0.2)

  for j in range (len(grades)):
    if j < 5:
      ax.annotate('', xy=(2*j+1, -0.11), xytext=(2*j, -0.11), annotation_clip=False, arrowprops=dict(arrowstyle='->', linewidth=2))
      ax.text(2*j+0.5, -0.15, grades[j], ha='center', fontsize=28)
    else:
      ax.annotate('', xy=(2*j+0.5, -0.11), xytext=(2*j, -0.11), annotation_clip=False, arrowprops=dict(arrowstyle='->', linewidth=2))
      ax.text(2*j+0.25, -0.15, grades[j], ha='center', fontsize=28)
    
    
  # set ticks
  y_ticks = np.arange(0, 120, step=20)/100  # Adjust as needed
  ax.set_yticks(y_ticks)
  ax.set_xticks(np.arange(0, len(semesters), 1), semesters)

  # Create a custom legend for the 5 trajectory groups 
  legend_elements_trajs = [Line2D([0], [0], marker='s', color='w', label=group, markerfacecolor=color, markersize=28) for group, color in zip(trajectory_groups, trajectory_colors)]

  # Create a custom legend for the summer break
  legend_element_summer = [Line2D([0], [0], marker='s', color='w', label='Summer Break', markerfacecolor='grey', markersize=28, alpha=0.3)]

  # Customize legend for k-means vs. GBTM
  legend_lines = [Line2D([0], [0], color='black', label='K-Means', linestyle='-', lw=2, marker='o', ms=10),
                  Line2D([0], [0], color='black', label='GBTM (Baseline)', linestyle='--', lw=2, marker='s', ms=10)]

  legend_elements = legend_elements_trajs + legend_element_summer + legend_lines
  legend1 = ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.7735, 0.5), fontsize=24)
  ax.add_artist(legend1)

  ax.tick_params(axis='both', which='both', labelsize=28)
  plt.tight_layout()

  plt.savefig("trajgps_kmeans_vs_gbtm.pdf", format="pdf")

  # Show the plot
  plt.show()
