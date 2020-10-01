---
toc: true
layout: post
description: How to use Kubeflow Pipelines to support HP Tuning using the Keras Tuner
categories: [ml, kfp, kubeflow, keras, tensorflow, hp_tuning]
title: Running a distributed Keras HP Tuning search using Kubeflow Pipelines
---


## Introduction

The performance of a machine learning model is often crucially dependent on the choice of good [hyperparameters][1]. For models of any complexity, relying on trial and error to find good values for these parameters does not scale. This tutorial shows how to use [Cloud AI Platform Pipelines][2]  in conjunction with [Keras Tuner][3] to build a hyperparameter-tuning workflow that uses distributed HP search.

[Cloud AI Platform Pipelines][4], currently in Beta, provides a way to deploy robust, repeatable machine learning pipelines along with monitoring, auditing, version tracking, and reproducibility, and gives you an easy-to-install, secure execution environment for your ML workflows. AI Platform Pipelines is based on [Kubeflow Pipelines][5] (KFP) installed on a [Google Kubernetes Engine (GKE)][6] cluster, and can run pipelines specified via both the KFP and TFX SDKs. See [this blog post][7] for more detail on the Pipelines tech stack.
You can create an AI Platform Pipelines installation with just a few clicks. After installing, you access AI Platform Pipelines by visiting the AI Platform Panel in the [Cloud Console][8].

[Keras Tuner][9] is a distributable hyperparameter optimization framework. Keras Tuner makes it easy to define a search space and leverage included algorithms to find the best hyperparameter values. It comes with several search  algorithms built-in, and is also designed to be easy for researchers to extend in order to experiment with new search algorithms.  It is straightforward to run the Keras Tuner in [distributed search mode][10], which we’ll leverage for this example.

The intent of a HP tuning search is typically not to do full training for each parameter combination, but to find the best starting points.  The number of epochs run in the HP search trials are typically smaller than that used in the full training. So, an HP tuning-based ML workflow could include:

- perform a distributed HP tuning search, and obtain the results 
- do concurrent model training runs for each of the best N parameter configurations, and export the model for each
- serve (some of) the resultant models (often after model evaluation).

As mentioned above, a Cloud AI Platform (KFP) Pipeline runs under the hood on a GKE cluster.  This makes it straightforward to implement this workflow— including the distributed HP search and model serving— so that you just need to launch a pipeline job to kick it off.  

This post highlights an example pipeline that does that. The example also shows how to use **preemptible** GPU-enabled VMS for the HP search, to reduce costs; and how to use [TF-serving][11] to deploy the trained model(s) on the same cluster for serving. As part of the process, we’ll see how GKE provides a scalable, resilient platform with easily-configured use of accelerators.

## About the dataset and modeling task

### The dataset

The [Cloud Public Datasets Program][12] makes available public datasets that are useful for experimenting with machine learning. Just as we did in our “[Explaining model predictions on structured data][13]” post, we’ll use data that is essentially a join of two public datasets stored in [BigQuery][14]: [London Bike rentals][15] and [NOAA weather data][16], with some additional processing to clean up outliers and derive additional GIS and day-of-week fields. 

### The modeling task and Keras model

We’ll use this dataset to build a [Keras][17] _regression model_ to predict the **duration** of a bike rental based on information about the start and end stations, the day of the week, the weather on that day, and other data. If we were running a bike rental company, for example, these predictions—and their explanations—could help us anticipate demand and even plan how to stock each location.

We’ll then use the Keras Tuner package to do an HP search using this model.

## Keras tuner in distributed mode on GKE with preemptible VMs

The [Keras Tuner][18] supports running a hyperparameter search  in [distributed mode][19]. 
[Google Kubernetes Engine (GKE)][20] makes it straightforward to configure and run a distributed HP tuning search.  GKE is  a good fit not only because it lets you easily distribute the HP tuning workload, but because you can leverage autoscaling to boost node pools for a large job, then scale down when the resources are no longer needed.  It’s also easy to deploy trained models for serving onto the same GKE cluster, using [TF-serving][21].  In addition, the Keras Tuner works well with [**preemptible VMs**][22], making it even cheaper to run your workloads. 

With the Keras Tuner’s distributed config, you specify one node as the ‘chief’, which coordinates the search, and ‘tuner’ nodes that do the actual work of running model training jobs using a given param set (the _trials_).  When you set up an HP search, you indicate the max number of trials to run, and how many _executions_ to run per trial. The Kubeflow Pipeline allows dynamic specification of the number of tuners to use for a given HP search— this determines how many trials you can run concurrently— as well as the max number of trials and number of executions.

We’ll define the tuner components as Kubernetes [_jobs_][23], each specified to have  1 _replica_.   This means that if a tuner job pod is terminated for some reason prior to job completion, Kubernetes will start up another replica.
Thus, the Keras Tuner’s HP search is a good fit for use of [preemptible VMs][24].  Because the HP search bookkeeping— orchestrated by the tuner `chief`, via an ‘oracle’ file— tracks the state of the trials,  the configuration is robust to a tuner pod terminating unexpectedly— say, due to a preemption— and a new one being restarted.  The new job pod will get its instructions from the ‘oracle’ and continue running _trials_.
The example uses GCS for the tuners’ shared file system.

Once the HP search has finished, any of the tuners can obtain information on the `N` best parameter sets (as well as export the best model(s)).  

## Defining the HP Tuning + training workflow as a pipeline

The definition of the pipeline itself is [here][25], specified using the KFP SDK.  It’s then compiled to an archive file and uploaded to AI Platforms Pipelines. (To compile it yourself, you’ll need to have the [KFP SDK installed][26]).  Pipeline steps are container-based, and you can find the Dockerfiles and underlying code for the steps under the example’s [`components`][27] directory.

The example pipeline first runs a distributed HP tuning search using a specified number of tuner workers,  then obtains the best `N` parameter sets—by default, it grabs the best two.  The pipeline step itself does not do the heavy lifting, but rather launches all the tuner [*jobs*][28] on GKE, which run concurrently, and monitors for their completion. (Unsurprisingly, this stage of the pipeline may run for quite a long time, depending upon how many HP search trials were specified and how many tuners are used for the distributed search).  

Concurrently to the Keras Tuner runs, the pipeline sets up a [TensorBoard visualization component][29], its log directory set to the GCS path under which the full training jobs are run.

The pipeline then runs full training jobs, concurrently, for each of the `N` best parameter sets.  It does this via the KFP [loop][30] construct, allowing the pipeline to support dynamic specification of `N`.  The training jobs can be monitored and compared using TensorBoard, both while they’re running and after they’ve completed.

Then, the trained models are deployed for serving for serving on the GKE cluster, using [TF-serving][31].  Each deployed model has its own cluster service endpoint.  
(While not included in this example, one could insert a step for model evaluation before making the decision about whether to deploy to TF-serving.)

For example, here is the [DAG][32] for a pipeline execution that did training and then deployed prediction services for the two best parameter configurations.

<figure>
<a href="https://storage.googleapis.com/amy-jo/images/kf-pls/pl_dag.png" target="_blank"><img src="https://storage.googleapis.com/amy-jo/images/kf-pls/pl_dag.png" width="60%"/></a>
<figcaption><br/><i>The DAG for keras tuner pipeline execution.  Here the two best parameter configurations were used for full training.</i></figcaption>
</figure>

## Running the example pipeline

> **Note**: this example may take a long time to run, and **incur significant charges** in its use of GPUs, depending upon how its parameters are configured.

To run the example, and for more detail on the KFP pipeline’s components, see the example’s [`README`][33].

## What’s next?…

One obvious next step in pipeline development would be to add components that evaluate each full model after training, before determining whether to deploy it, e.g. using [TensorFlow Model Analysis][34]  (TFMA).  Stay tuned for a follow-up blog post that explores how to do that. 

[1]:	https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)
[2]:	https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-ai-platform-pipelines
[3]:	https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html
[4]:	https://cloud.google.com/ai-platform/pipelines/docs
[5]:	https://www.kubeflow.org/docs/pipelines/
[6]:	https://cloud.google.com/kubernetes-engine
[7]:	https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-ai-platform-pipelines
[8]:	https://console.cloud.google.com/
[9]:	https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html
[10]:	https://keras-team.github.io/keras-tuner/tutorials/distributed-tuning/
[11]:	https://www.tensorflow.org/tfx/guide/serving
[12]:	https://cloud.google.com/bigquery/public-data/
[13]:	https://cloud.google.com/blog/products/ai-machine-learning/explaining-model-predictions-structured-data
[14]:	https://cloud.google.com/bigquery/
[15]:	https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=london_bicycles&page=dataset
[16]:	https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=noaa_gsod&page=dataset
[17]:	https://keras.io/
[18]:	https://keras-team.github.io/keras-tuner/
[19]:	https://keras-team.github.io/keras-tuner/tutorials/distributed-tuning/
[20]:	https://cloud.google.com/kubernetes-engine
[21]:	https://www.tensorflow.org/tfx/guide/serving
[22]:	https://cloud.google.com/kubernetes-engine/docs/how-to/preemptible-vms
[23]:	https://kubernetes.io/docs/concepts/workloads/controllers/job/
[24]:	https://cloud.google.com/preemptible-vms
[25]:	https://github.com/amygdala/code-snippets/blob/keras_tuner2/ml/kubeflow-pipelines/keras_tuner/example_pipelines/bw_ktune.py
[26]:	https://www.kubeflow.org/docs/pipelines/sdk/install-sdk/#install-the-kubeflow-pipelines-sdk
[27]:	https://github.com/amygdala/code-snippets/tree/keras_tuner2/ml/kubeflow-pipelines/keras_tuner/components
[28]:	https://cloud.google.com/kubernetes-engine/docs/how-to/jobs
[29]:	https://github.com/kubeflow/pipelines/blob/master/components/tensorflow/tensorboard/prepare_tensorboard/component.yaml
[30]:	https://github.com/kubeflow/pipelines/tree/master/samples/core/loop_parameter
[31]:	https://www.tensorflow.org/tfx/guide/serving
[32]:	https://en.wikipedia.org/wiki/Directed_acyclic_graph
[33]:	https://github.com/amygdala/code-snippets/blob/keras_tuner2/ml/kubeflow-pipelines/keras_tuner/README.md
[34]:	https://www.tensorflow.org/tfx/model_analysis/get_started

