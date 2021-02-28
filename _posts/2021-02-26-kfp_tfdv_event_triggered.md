---
toc: true
layout: post
description: how to set up event-triggered Kubeflow Pipelines runs, and use TFDV to detect data drift
categories: [ml, kfp, mlops, tfdv, gcf]
title: Event-triggered Kubeflow Pipeline runs, and using TFDV to detect data drift
---

## Introduction

With ML workflows, it is often insufficient to train and deploy a given model just once.  Even if the model has desired accuracy initially, this can change if the data used for making prediction requests becomes— perhaps over time— sufficiently different from the data used to originally train the model.

When new data becomes available, which could be used for retraining a model, it can be helpful to apply techniques for analyzing _data ‘drift’_, and determining whether the drift is sufficiently anomalous to warrant retraining yet.
It can also be useful to trigger such an analysis— and potential re-run of your training pipeline— _automatically_, upon arrival of new data.

This blog post highlights an [example notebook][1] that shows how to set up such a scenario with [Kubeflow Pipelines][2] (KFP).
It shows how to build a pipeline that checks for statistical drift across successive versions of a dataset and uses that information to make a decision on whether to (re)train a model[^1]; and how to configure event-driven deployment of pipeline jobs when new data arrives.

The notebook builds on an example highlighted in a [previous blog post][3] — which shows a KFP training and serving pipeline— and introduces two primary new concepts:

-  the example demonstrates use of the [TensorFlow Data Validation (TFDV)][4] library to build pipeline components that derive **dataset statistics** and detect **drift** between older and newer dataset versions, and shows how to use drift information to decide whether to retrain a model on newer data.
-  the example shows how to support **event-triggered** launch of Kubeflow Pipelines runs from a [Cloud Functions][5] (GCF) function, where the Function run is triggered by addition of a file to a given [Cloud Storage][6] (GCS) bucket.

The machine learning task uses a tabular dataset that joins London bike rental information with weather data, and train a Keras model to predict rental duration. See [this][7] and [this][8] blog post and associated [README][9] for more background on the dataset and model architecture.

<figure>
<a href="https://storage.googleapis.com/amy-jo/images/kf-pls/CleanShot%202021-02-26%20at%2011.30.13%402x.png" target="_blank"><img src="https://storage.googleapis.com/amy-jo/images/kf-pls/CleanShot%202021-02-26%20at%2011.30.13%402x.png" width="60%"/></a>
<figcaption><br/><i>A pipeline run using TFDV-based components to detect 'data drift'.</i></figcaption>
</figure>

## Running the example notebook

The [example notebook][10] requires a [Google Cloud Platform (GCP)][11] account and project, ideally with quota for using GPUs, and— as detailed in the notebook— an installation of **AI Platform Pipelines (Hosted Kubeflow Pipelines)** (that is, an installation of KFP on [Google Kubernetes Engine (GKE)][12]), with a few additional configurations once installation is complete.

The notebook can be run using either [Colab][13] ([open directly](https://colab.research.google.com/github/amygdala/code-snippets/blob/master/ml/notebook_examples/hosted_kfp/event_triggered_kfp_pipeline_bw.ipynb)) or [AI Platform Notebooks][14] ([open directly](https://console.cloud.google.com/ai-platform/notebooks/deploy-notebook?download_url=https://raw.githubusercontent.com/amygdala/code-snippets/master/ml/notebook_examples/hosted_kfp/event_triggered_kfp_pipeline_bw.ipynb)).

## Creating TFDV-based KFP components

Our first step is to build the TFDV components that we want to use in our pipeline.

> Note: For this example, our training data is in GCS, in CSV-formatted files.  So, we can take advantage of TFDV’s ability to process CSV files.  The TFDV libraries can also process files in `TFRecords` format.

We'll define both TFDV KFP pipeline *components* as ['lightweight' Python-function-based components][15]. For each component, we define a function, then call `kfp.components.func_to_container_op()` on that function to build a **reusable** component in `.yaml` format.
Let’s take a closer look at how this works (details are in the [notebook][16]).

Below is the Python function we’ll use to generate TFDV statistics from a collection of `csv` files.  The function— and the component we’ll create from it— outputs the path to the generated stats file.  When we define a pipeline that uses this component, we’ll use this step’s output as input to another pipeline step.
TFDV uses a [Beam][17] pipeline— not to be confused with KFP Pipelines— to implement the stats generation. Depending upon configuration, the component can use either the Direct (local) runner or the [Dataflow][18] runner.  Running the Beam pipeline on Dataflow rather than locally can make sense with large datasets.


```python
from typing import NamedTuple

def generate_tfdv_stats(input_data: str, output_path: str, job_name: str, use_dataflow: str,
                        project_id: str, region:str, gcs_temp_location: str, gcs_staging_location: str,
                        whl_location: str = '', requirements_file: str = 'requirements.txt'
) -> NamedTuple('Outputs', [('stats_path', str)]):

  import logging
  import time

  import tensorflow_data_validation as tfdv
  import tensorflow_data_validation.statistics.stats_impl
  from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions, StandardOptions, SetupOptions

  logging.getLogger().setLevel(logging.INFO)
  logging.info("output path: %s", output_path)
  logging.info("Building pipeline options")
  # Create and set your PipelineOptions.
  options = PipelineOptions()

  if use_dataflow == 'true':
    logging.info("using Dataflow")
    if not whl_location:
      logging.warning('tfdv whl file required with dataflow runner.')
      exit(1)
    google_cloud_options = options.view_as(GoogleCloudOptions)
    google_cloud_options.project = project_id
    google_cloud_options.job_name = '{}-{}'.format(job_name, str(int(time.time())))
    google_cloud_options.staging_location = gcs_staging_location
    google_cloud_options.temp_location = gcs_temp_location
    google_cloud_options.region = region
    options.view_as(StandardOptions).runner = 'DataflowRunner'

    setup_options = options.view_as(SetupOptions)
    setup_options.extra_packages = [whl_location]
    setup_options.requirements_file = 'requirements.txt'

  tfdv.generate_statistics_from_csv(
    data_location=input_data, output_path=output_path,
    pipeline_options=options)

  return (output_path, )
```

To turn this function into a KFP _component_, we’ll call `kfp.components.func_to_container_op()`.  We’re passing it a base container image to use: `gcr.io/google-samples/tfdv-tests:v1`.  This base image has the TFDV libraries already installed, so that we don’t need to install them ‘inline’ when we run a pipeline step based on this component.

```python
import kfp
kfp.components.func_to_container_op(generate_tfdv_stats,
    output_component_file='tfdv_component.yaml', base_image='gcr.io/google-samples/tfdv-tests:v1')
```

We’ll take the same approach to build a second TFDV-based component, one which detects _drift_ between datasets by comparing their stats.  The TFDV library makes this straightforward.  We’re using a drift comparator appropriate for a _regression_ model— as used in the example pipeline— and looking for drift on a given set of fields (in this case, for example purposes, just one).
The `tensorflow_data_validation.validate_statistics()` call will then tell us whether the drift anomaly for that field is over the specified threshold. See the [TFDV docs][19] for more detail.

```python
  schema1 = tfdv.infer_schema(statistics=stats1)
  tfdv.get_feature(schema1, 'duration').drift_comparator.jensen_shannon_divergence.threshold = 0.01
  drift_anomalies = tfdv.validate_statistics(
      statistics=stats2, schema=schema1, previous_statistics=stats1)
```

(The details of this second component definition are in the example notebook).

##  Defining a pipeline that uses the TFDV components

After we’ve defined both TFDV components— one to generate stats for a dataset, and one to detect drift between datasets— we’re ready to build a Kubeflow Pipeline that uses these components, in conjunction with previously-built components for a training & serving workflow.

### Instantiate pipeline _ops_ from the components

KFP components in `yaml` format are shareable and reusable.  We’ll build our pipeline by starting with some already-built components— (described in more detail [here][20])— that support our basic ‘train/evaluate/deploy’ workflow.

We’ll instantiate some pipeline ops from these pre-existing components like this, by loading them via URL:

```python
import kfp.components as comp

# pre-existing components
train_op = comp.load_component_from_url(
  'https://raw.githubusercontent.com/amygdala/code-snippets/master/ml/kubeflow-pipelines/keras_tuner/components/train_component.yaml'
  )
... etc. ...
```

… then create our TFDV ops from the new components we just built:

```python

tfdv_op = comp.load_component_from_file(
  'tfdv_component.yaml'
  )
tfdv_drift_op = comp.load_component_from_file(
  'tfdv_drift_component.yaml'
  )
```

Then, we define a KFP pipeline from the defined ops.  We’re not showing the pipeline in full here— see the notebook for details.
Two pipeline steps use the `tfdv_op`, which generates the stats.  `tfdv1` generates stats for the test data, and `tfdv2` for the training data.
In the following, you can see that the `tfdv_drift` step takes as input the output from the `tfdv2` (stats for training data) step.


```python
@dsl.pipeline(
  name='bikes_weather_tfdv',
  description='Model bike rental duration given weather'
)
def bikes_weather_tfdv(
  ... other pipeline params ...
  working_dir: str = 'gs://YOUR/GCS/PATH',
  data_dir: str = 'gs://aju-dev-demos-codelabs/bikes_weather/',
  project_id: str = 'YOUR-PROJECT-ID',
  region: str = 'us-central1',
  requirements_file: str = 'requirements.txt',
  job_name: str = 'test',
  whl_location: str = 'tensorflow_data_validation-0.26.0-cp37-cp37m-manylinux2010_x86_64.whl',
  use_dataflow: str = '',
  stats_older_path: str = 'gs://aju-dev-demos-codelabs/bikes_weather_chronological/evaltrain1.pb'
  ):
  ...

  tfdv1 = tfdv_op(  # TFDV stats for the test data
    input_data='%stest-*.csv' % (data_dir,),
    output_path='%s/tfdv_expers/%s/eval/evaltest.pb' % (working_dir, dsl.RUN_ID_PLACEHOLDER),
    job_name='%s-1' % (job_name,),
    use_dataflow=use_dataflow,
    project_id=project_id, region=region,
    gcs_temp_location='%s/tfdv_expers/tmp' % (working_dir,),
    gcs_staging_location='%s/tfdv_expers' % (working_dir,),
    whl_location=whl_location, requirements_file=requirements_file
    )
  tfdv2 = tfdv_op(  # TFDV stats for the training data
    input_data='%strain-*.csv' % (data_dir,),
    # output_path='%s/%s/eval/evaltrain.pb' % (output_path, dsl.RUN_ID_PLACEHOLDER),
    output_path='%s/tfdv_expers/%s/eval/evaltrain.pb' % (working_dir, dsl.RUN_ID_PLACEHOLDER),
    job_name='%s-2' % (job_name,),
    use_dataflow=use_dataflow,
    project_id=project_id, region=region,
    gcs_temp_location='%s/tfdv_expers/tmp' % (working_dir,),
    gcs_staging_location='%s/tfdv_expers' % (working_dir,),
    whl_location=whl_location, requirements_file=requirements_file
    )

  # compare generated training data stats with stats from a previous version
  # of the training data set.
  tfdv_drift = tfdv_drift_op(stats_older_path, tfdv2.outputs['stats_path'])

  # proceed with training if drift is detected (or if no previous stats were provided)
  with dsl.Condition(tfdv_drift.outputs['drift'] == 'true'):
    train = train_op(...)
    eval_metrics = eval_metrics_op(...)
    with dsl.Condition(eval_metrics.outputs['deploy'] == 'deploy'):
      serve = serve_op(...)

```

While not all pipeline details are shown, you can see that this pipeline definition includes some conditional expressions; parts of the pipeline will run only if an output of an ‘upstream’ step meets the given conditions.  We start the model training step if drift anomalies were detected.  (And, once training is completed, we’ll deploy the model for serving only if its evaluation metrics meet certain thresholds).

Here’s the [DAG][21] for this pipeline.  You can see the conditional expressions reflected; and can see that the step to generates stats for the test dataset provides no downstream dependencies, but the stats on the training set are used as input for the drift detection step.

<figure>
<a href="https://storage.googleapis.com/amy-jo/images/kf-pls/bw_tfdv_pipeline.png" target="_blank"><img src="https://storage.googleapis.com/amy-jo/images/kf-pls/bw_tfdv_pipeline.png" width="50%"/></a>
<figcaption><br/><i>The pipeline DAG</i></figcaption>
</figure>

Here’s a pipeline run in progress:

<figure>
<a href="https://storage.googleapis.com/amy-jo/images/kf-pls/bw_tfdv_pipeline_run.png" target="_blank"><img src="https://storage.googleapis.com/amy-jo/images/kf-pls/bw_tfdv_pipeline_run.png" width="60%"/></a>
<figcaption><br/><i>A pipeline run in progress.</i></figcaption>
</figure>

See the [example notebook][22] for more details on how to run this pipeline.

## Event-triggered pipeline runs

Once you have defined this pipeline, a next useful step is to automatically run it when an update to the dataset is available, so that each dataset update triggers an analysis of data drift and potential model (re)training.

We’ll show how to do this using [Cloud Functions (GCF)][23], by setting up a function that is triggered when new data is added to a [GCS][24] bucket.

### Set up a GCF function to trigger a pipeline run when a dataset is updated

We’ll define and deploy a [Cloud Functions (GCF)][25] function that launches a run of this pipeline when new training data becomes available, as triggered by the creation or modification of a file in a ‘trigger’ bucket on GCS.

In most cases, you don’t want to launch a new pipeline run for every new file added to a dataset— since typically, the dataset will be comprised of a collection of files, to which you will add/update multiple files in a batch. So, you don’t want the ‘trigger bucket’ to be the dataset bucket (if the data lives on GCS)— that will trigger unwanted pipeline runs.
Instead, we’ll trigger a pipeline run after the upload of a _batch_ of new data has completed.

To do this, we’ll use an approach where the the 'trigger' bucket is different from the bucket used to store dataset files. ‘Trigger files’ uploaded to that bucket are expected to contain the path of the updated dataset as well as the path to the data stats file generated for the last model trained.
A trigger file is uploaded once the new data upload has completed, and that upload triggers a run of the GCF function, which in turn reads info on the new data path from the trigger file and launches the pipeline job.

#### Define the GCF function

To set up this process, we’ll first define the GCF function in a file called `main.py`, as well as an accompanying requirements file in the same directory that specifies the libraries to load prior to running the function.  The requirements file will indicate to install the KFP SDK:
```python
kfp==1.4
```

The code looks like this (with some detail removed); we parse the trigger file contents and use that information to launch a pipeline run. The code uses the values of several environment variables that we will set when uploading the GCF function.

```python
import logging
import os

import kfp
from kfp import dsl
from kfp import compiler
from kfp import components

from google.cloud import storage

PIPELINE_PROJECT_ID = os.getenv('PIPELINE_PROJECT_ID')
...etc...

def read_trigger_file(data, context, storage_client):
    """Read the contents of the trigger file and return as string.
    """
    ....
    bucket = storage_client.get_bucket(data['bucket'])
    blob = bucket.get_blob(data['name'])
    trigger_file_string = blob.download_as_string().strip()
    logging.info('trigger file contents: {}'.format(trigger_file_string))
    return trigger_file_string.decode('UTF-8')


def gcs_update(data, context):
    """Background Cloud Function to be triggered by Cloud Storage.
    """

    storage_client = storage.Client()
    # get the contents of the trigger file
    trigger_file_string = read_trigger_file(data, context, storage_client)
    trigger_file_info = trigger_file_string.strip().split('\n')
    # then run the pipeline using the given job spec, passing the trigger file contents
    # as parameter values.
    logging.info('running pipeline with id %s...', PIPELINE_ID)
    # create the client object
    client = kfp.Client(host=PIPELINE_HOST)
    # deploy the pipeline run
    run = client.run_pipeline(EXP_ID, 'bw_tfdv_gcf', pipeline_id=PIPELINE_ID,
                          params={'working_dir': WORKING_DIR,
                                  'project_id': PIPELINE_PROJECT_ID,
                                  'use_dataflow': USE_DATAFLOW,
                                  'data_dir': trigger_file_info[0],
                                  'stats_older_path': trigger_file_info[1]})

    logging.info('job response: %s', run)
```

Then we’ll deploy the GCF function as follows. Note that we’re indicating to use the `gcs_update` definition (from `main.py`), and specifying the trigger bucket.   Note also how we're setting environment vars as part of the deployment.

```bash
gcloud functions deploy gcs_update --set-env-vars \
  PIPELINE_PROJECT_ID={PROJECT_ID},WORKING_DIR={WORKING_DIR},PIPELINE_SPEC={PIPELINE_SPEC},PIPELINE_ID={PIPELINE_ID},PIPELINE_HOST={PIPELINE_HOST},EXP_ID={EXP_ID},USE_DATAFLOW=true \
  --runtime python37 --trigger-resource {TRIGGER_BUCKET} --trigger-event google.storage.object.finalize
```

### Trigger a pipeline run when new data becomes available

Once the GCF function is set up, it will run when a file is added to (or modified in) the trigger bucket.  For this simple example, the GCF function expects trigger files of the following format, where the first line is the path to the updated dataset, and the second line is the path to the TFDV stats for the dataset used for the previously-trained model.  More generally, such a trigger file can contain whatever information is necessary to determine how to parameterize the pipeline run.

```bash
gs://path/to/new/or/updated/dataset/
gs://path/to/stats/from/previous/dataset/stats.pb
```


## Summary

This blog post showed how to build Kubeflow Pipeline components, using the TFDV libraries, to analyze datasets and detect data drift.  Then, it showed how to support event-triggered pipeline runs via GCF.


[^1]:	In this example, we show full model retraining on a new dataset.  An alternate scenario— not covered here— could involve _tuning_ an existing model with new data.

[1]:	https://github.com/amygdala/code-snippets/blob/master/ml/notebook_examples/hosted_kfp/event_triggered_kfp_pipeline_bw.ipynb
[2]:	https://www.kubeflow.org/docs/pipelines/
[3]:	https://amygdala.github.io/gcp_blog/ml/kfp/mlops/keras/hp_tuning/2020/10/26/metrics_eval_component.html
[4]:	https://www.tensorflow.org/tfx/guide/tfdv
[5]:	https://cloud.google.com/functions/docs/
[6]:	https://cloud.google.com/storage
[7]:	https://amygdala.github.io/gcp_blog/ml/kfp/kubeflow/keras/tensorflow/hp_tuning/2020/10/19/keras_tuner.html
[8]:	https://amygdala.github.io/gcp_blog/ml/kfp/mlops/keras/hp_tuning/2020/10/26/metrics_eval_component.html
[9]:	https://github.com/amygdala/code-snippets/blob/master/ml/kubeflow-pipelines/keras_tuner/README.md
[10]:	https://github.com/amygdala/code-snippets/blob/master/ml/notebook_examples/hosted_kfp/event_triggered_kfp_pipeline_bw.ipynb
[11]:	https://cloud.google.com/
[12]:	https://cloud.google.com/kubernetes-engine
[13]:	https://colab.research.google.com/
[14]:	https://cloud.google.com/ai-platform-notebooks
[15]:	https://www.kubeflow.org/docs/pipelines/sdk/python-function-components/
[16]:	https://github.com/amygdala/code-snippets/blob/master/ml/notebook_examples/hosted_kfp/event_triggered_kfp_pipeline_bw.ipynb
[17]:	https://beam.apache.org/
[18]:	https://cloud.google.com/dataflow#section-5
[19]:	https://www.tensorflow.org/tfx/data_validation/get_started#checking_data_skew_and_drift
[20]:	https://github.com/amygdala/code-snippets/blob/master/ml/kubeflow-pipelines/keras_tuner/README.md
[21]:	https://en.wikipedia.org/wiki/Directed_acyclic_graph
[22]:	https://github.com/amygdala/code-snippets/blob/master/ml/notebook_examples/hosted_kfp/event_triggered_kfp_pipeline_bw.ipynb
[23]:	https://cloud.google.com/functions/docs/
[24]:	https://cloud.google.com/storage
[25]:	https://cloud.google.com/functions/docs/

