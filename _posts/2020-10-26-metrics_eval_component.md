---
toc: true
layout: post
description: how to create a Kubeflow Pipelines component from a python function, and define and deploy pipelines from a notebook
categories: [ml, kfp, mlops, keras, hp_tuning]
title: Keras Tuner KFP example, part II— creating a lightweight component for metrics evaluation
---

## Introduction

This [blog post](https://amygdala.github.io/gcp_blog/ml/kfp/kubeflow/keras/tensorflow/hp_tuning/2020/10/19/keras_tuner.html) and accompanying [tutorial](https://github.com/amygdala/code-snippets/blob/master/ml/kubeflow-pipelines/keras_tuner/README.md) walked through how to build a [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/) (KFP) pipeline that uses the [Keras Tuner](https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html) to build a hyperparameter-tuning workflow that uses distributed HP search.

That pipeline does HP tuning, then runs full training on the N best parameter sets identified from the HP search, then deploys the full models to [TF-serving](https://www.tensorflow.org/tfx/guide/serving).  
One thing that was missing from that pipeline was any check on the quality of the trained models prior to deployment to TF-Serving.

This post is based on an [example notebook](https://github.com/amygdala/code-snippets/blob/master/ml/kubeflow-pipelines/keras_tuner/notebooks/metrics_eval_component.ipynb), and is a follow-on to that tutorial.  You can follow along in the notebook instead if you like (see below).

Here, we'll show how you can create a KFP "lightweight component", built from a python function, to do a simple threshold check on some of the model metrics in order to decide whether to deploy the model. (This is a pretty simple approach, that we're using for illustrative purposes; for production models you'd probably want to do more sophisticated analyses. The [TFMA library](https://www.tensorflow.org/tfx/model_analysis/get_started) might be of interest).
We'll also show how to use the KFP SDK to define and run pipelines from a notebook.


## Setup

This example assumes that you've **done the setup indicated in the [README](https://github.com/amygdala/code-snippets/blob/master/ml/kubeflow-pipelines/keras_tuner/README.md)**, and have an AI Platform Pipelines (Hosted KFP) installation, with GPU node pools added to the cluster.

### Create an AI Platform Notebooks instance


In addition, create an AI Platform Notebooks instance on which to run [the example notebook](https://github.com/amygdala/code-snippets/blob/master/ml/kubeflow-pipelines/keras_tuner/notebooks/metrics_eval_component.ipynb) on which this post is based. See setup instructions [here](https://cloud.google.com/ai-platform/notebooks/docs). (You can run this notebook in other environments, including, locally, but that requires additional auth setup that we won't go into here).

**Once your notebook instance is set up, you should be able to use [this link](https://console.cloud.google.com/ai-platform/notebooks/deploy-notebook?name=Create%20a%20new%20KFP%20component%20from%20a%20notebook&download_url=https%3A%2F%2Fraw.githubusercontent.com%2Famygdala%2Fcode-snippets%2Fmaster%2Fml%2Fkubeflow-pipelines%2Fkeras_tuner%2Fnotebooks%2Fmetrics_eval_component.ipynb&url=https%3A%2F%2Fgithub.com%2Famygdala%2Fcode-snippets%2Fblob%2Fmaster%2Fml%2Fkubeflow-pipelines%2Fkeras_tuner%2Fnotebooks%2Fmetrics_eval_component.ipynb) to upload and run the notebook.**

### Install the KFP SDK

Next, we'll install the KFP SDK.  In a notebook, you may need to restart the kernel so it's available for import.


```python
!pip install --user -U kfp kfp-server-api
```


Next, we'll do some imports:


```python
import kfp  # the Pipelines SDK. 
from kfp import compiler
import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.components as comp
```

## Defining a new 'lightweight component' based on a python function

'Lightweight' KFP python components allow you to create a component from a python function definition, and do not require you to build a new container image for every code change. They're helpful for fast iteration in a notebook environment. You can read more [here](https://github.com/kubeflow/pipelines/blob/master/samples/core/lightweight_component/lightweight_component.ipynb).

In this section, we'll create a lightweight component that uses training metrics info to decide whether to deploy a model.
We'll pass a "threshold" dict as a component arg, and compare those thresholds to the metrics values, and use that info to decide whether or not to deploy.  Then we'll output a string indicating the decision. 

(As mentioned above, for production models you'd probably want to do a more substantial analysis. The [TFMA library](https://www.tensorflow.org/tfx/model_analysis/get_started) might be of interest. Stay tuned for a follow-on post that uses TFMA).

Then we'll define a pipeline that uses the new component. In the pipeline spec, we'll make the 'serve' step conditional on the "metrics" op output.

First, we'll define the component function, `eval_metrics`:


```python
from typing import NamedTuple

def eval_metrics(
  metrics: str,
  thresholds: str
) -> NamedTuple('Outputs', [('deploy', str)]):

  import json
  import logging

  def regression_threshold_check(metrics_info):
    for k, v in thresholds_dict.items():
      logging.info('k {}, v {}'.format(k, v))
      if k in ['root_mean_squared_error', 'mae']:
        if metrics_info[k][-1] > v:
          logging.info('{} > {}; returning False'.format(metrics_info[k][0], v))
          return ('False', )
    return ('deploy', )

  logging.getLogger().setLevel(logging.INFO)

  thresholds_dict = json.loads(thresholds)
  logging.info('thresholds dict: {}'.format(thresholds_dict))
  logging.info('metrics: %s', metrics)
  metrics_dict = json.loads(metrics)

  logging.info("got metrics info: %s", metrics_dict)
  res = regression_threshold_check(metrics_dict)
  logging.info('deploy decision: %s', res)
  return res

```

To keep things simple, we're comparing only RMSE and MAE with given threshold values.  (This function is tailored for our Keras regression model). Lower is better, so if a threshold value is higher than the associated model metric, we won't deploy. 

Next, we'll create a 'container op' from the `eval_metrics` function definition, via the `funct_to_container_op` method. As one of the method args, we specify the base container image that will run the function. 
Here, we're using one of the [Deep Learning Container images](https://cloud.google.com/ai-platform/deep-learning-containers/docs/).  (This container image installs more than is necessary for this simple function, but these DL images can be useful for many ML-related components).


```python
eval_metrics_op = comp.func_to_container_op(eval_metrics, base_image='gcr.io/deeplearning-platform-release/tf2-cpu.2-3:latest')
```

## Define a pipeline that uses the new "metrics" op

Now, we can define a new pipeline that uses the new op and makes the model serving conditional on the results. 

The new `eval_metrics_op` takes as an input one of the `train_op` outputs, which outputs a final metrics dict. (We "cheated" a bit, as the training component was already designed to output this info; in other cases you might end up defining a new version of such an op that outputs the new info you need).

Then, we'll wrap the serving op in a *conditional*; we won't set up a TF-serving service unless the `eval_metrics` op has certified that it is okay.

Note that this new version of the pipeline also has a new input parameter— the `thresholds` dict.

To keep things simple, we'll first define a pipeline that skips the HP tuning part of the pipeline used [here](https://github.com/amygdala/code-snippets/blob/master/ml/kubeflow-pipelines/keras_tuner/README.md).  This will make it easier to test your new op with a pipeline that takes a shorter time to run.

Then in a following section we'll show how to augment the full HP tuning pipeline to include the new op.


We'll first instantiate the other pipeline ops from their [reusable components](https://www.kubeflow.org/docs/pipelines/sdk/component-development/) definitions.  (And we've defined the `eval_metrics_op` above).


```python

train_op = comp.load_component_from_url(
  'https://raw.githubusercontent.com/amygdala/code-snippets/master/ml/kubeflow-pipelines/keras_tuner/components/train_component.yaml'
  )
serve_op = comp.load_component_from_url(
  'https://raw.githubusercontent.com/amygdala/code-snippets/master/ml/kubeflow-pipelines/keras_tuner/components/serve_component.yaml'
  )

tb_op = comp.load_component_from_url(
  'https://raw.githubusercontent.com/kubeflow/pipelines/master/components/tensorflow/tensorboard/prepare_tensorboard/component.yaml' 
  )
```

Next, we'll define the pipeline itself.  You might notice that this pipeline has a new parameter, `thresholds`.

This pipeline first sets up a TensorBoard visualization for monitoring the training run. Then it starts the training. Once training is finished, the new op checks whether the trained model's final metrics are above the given threshold(s). 
If so (using the KFP `dsl.Condition` construct), TF-serving is used to set up a prediction service on the Pipelines GKE cluster.

You can see that data is being passed between the pipeline ops. [Here's a tutorial](https://gist.github.com/amygdala/bfa0f599a4814b3261367f558a852bfe) that goes into how that works in more detail.


```python
@dsl.pipeline(
  name='bikes_weather_metrics',
  description='Model bike rental duration given weather'
)
def bikes_weather_metrics( 
  train_epochs: int = 2,
  working_dir: str = 'gs://YOUR/GCS/PATH',  # for the full training jobs
  data_dir: str = 'gs://aju-dev-demos-codelabs/bikes_weather/',
  steps_per_epoch: int = -1 ,  # if -1, don't override normal calcs based on dataset size
  hptune_params: str = '[{"num_hidden_layers": %s, "learning_rate": %s, "hidden_size": %s}]' % (3, 1e-2, 64),
  thresholds: str = '{"root_mean_squared_error": 2000}'
  ):

  # create TensorBoard viz for the parent directory of all training runs, so that we can
  # compare them.
  tb_viz = tb_op(
    log_dir_uri='%s/%s' % (working_dir, dsl.RUN_ID_PLACEHOLDER)
  )

  train = train_op(
    data_dir=data_dir,
    workdir='%s/%s' % (tb_viz.outputs['log_dir_uri'], 0),
    tb_dir=tb_viz.outputs['log_dir_uri'],
    epochs=train_epochs, steps_per_epoch=steps_per_epoch,
    hp_idx=0, 
    hptune_results=hptune_params
    )

  eval_metrics = eval_metrics_op(
    thresholds=thresholds,
    metrics=train.outputs['metrics_output_path'],
    )

  with dsl.Condition(eval_metrics.outputs['deploy'] == 'deploy'):  # conditional serving
    serve = serve_op(
      model_path=train.outputs['train_output_path'],
      model_name='bikesw',
      namespace='default'
      )
  train.set_gpu_limit(2)
```

Now we can run the pipeline from the notebook.  First create a client object to talk to your KFP installation. Using that client, create (or get) an _Experiment_ (which lets you create semantic groupings of pipeline runs).

You'll need to set the correct host endpoint for your pipelines installation when you create the client.  Visit the [Pipelines panel in the Cloud Console](https://console.cloud.google.com/ai-platform/pipelines/clusters) and click on the **SETTINGS** gear for the desired installation to get its endpoint.


```python
# CHANGE THIS with the info for your KFP cluster installation
client = kfp.Client(host='xxxxxxxx-dot-us-centralx.pipelines.googleusercontent.com')
```


```python
exp = client.create_experiment(name='bw_expers')  # this is a 'get or create' call
```

(If the `create_experiment` call failed, double check your host endpoint value).

Now, we can compile and then run the pipeline.  We'll set some vars with pipeline params:


```python
WORKING_DIR = 'gs://YOUR_GCS/PATH'
TRAIN_EPOCHS = 2
```

Now we'll compile and run the pipeline.  

Note that this pipeline is configured to use a GPU node for the training step, so make sure that you have set up a GPU node pool for the cluster that your KFP installation is running on, as described in this [README](https://github.com/amygdala/code-snippets/blob/master/ml/kubeflow-pipelines/keras_tuner/README.md). Note also that GPU nodes are more expensive.  
If you want, you can comment out the `train.set_gpu_limit(2)` line in the pipeline definition above to run training on a CPU node.


```python
compiler.Compiler().compile(bikes_weather_metrics, 'bikes_weather_metrics.tar.gz')
```


```python
run = client.run_pipeline(exp.id, 'bw_metrics_test', 'bikes_weather_metrics.tar.gz',
                          params={'working_dir': WORKING_DIR, 'train_epochs': TRAIN_EPOCHS
                                 # 'thresholds': THRESHOLDS
                                 })
```

Once you've kicked off the run, click the generated link to see the pipeline run in the Kubeflow Pipelines dashboard of your pipelines installation. (See the last section for more info on how to use your trained and deployed model for prediction).

**Note**: It's also possible to start a pipeline run directly from the pipeline function definition, skipping the local compilation, like this:
```python
kfp.Client(host=kfp_endpoint).create_run_from_pipeline_func(<pipeline_function_name>, arguments={})
```

## Use the new "metrics" op with the full Keras Tuner pipeline

To keep things simple, the pipeline above didn't do an HP tuning search.
Below is how the full pipeline from [this tutorial](https://github.com/amygdala/code-snippets/blob/master/ml/kubeflow-pipelines/keras_tuner/README.md) would be redefined to use this new op.  

This definition assumes that you've run the cells above that instantiated the ops from their component specs. This new definition includes an additional `hptune` op (defined "inline" using `dsl.ContainerOp()`) that deploys the distributed HP tuning job and then waits for the results.  

> **Important note**: this example may take a long time to run, and **incur significant charges** in its use of GPUs, depending upon how its parameters are configured.


```python
@dsl.pipeline(
  name='bikes_weather_keras_tuner',
  description='Model bike rental duration given weather, use Keras Tuner'
)
def bikes_weather_hptune(
  tune_epochs: int = 2,
  train_epochs: int = 5,
  num_tuners: int = 8,
  bucket_name: str = 'YOUR_BUCKET_NAME',  # used for the HP dirs; don't include the 'gs://'
  tuner_dir_prefix: str = 'hptest',
  tuner_proj: str = 'p1',
  max_trials: int = 128,
  working_dir: str = 'gs://YOUR/GCS/PATH',  # for the full training jobs
  data_dir: str = 'gs://aju-dev-demos-codelabs/bikes_weather/',
  steps_per_epoch: int = -1 ,  # if -1, don't override normal calcs based on dataset size
  num_best_hps: int = 2,  # the N best parameter sets for full training
  # the indices to the best param sets; necessary in addition to the above param because of
  # how KFP loops work currently.  Must be consistent with the above param.
  num_best_hps_list: list = [0, 1],
  thresholds: str = '{"root_mean_squared_error": 2000}'
  ):

  hptune = dsl.ContainerOp(
      name='ktune',
      image='gcr.io/google-samples/ml-pipeline-bikes-dep:b97ee76',
      arguments=['--epochs', tune_epochs, '--num-tuners', num_tuners,
          '--tuner-dir', '%s/%s' % (tuner_dir_prefix, dsl.RUN_ID_PLACEHOLDER),
          '--tuner-proj', tuner_proj, '--bucket-name', bucket_name, '--max-trials', max_trials,
          '--namespace', 'default', '--num-best-hps', num_best_hps, '--executions-per-trial', 2,
          '--deploy'
          ],
      file_outputs={'hps': '/tmp/hps.json'},
      )

  # create TensorBoard viz for the parent directory of all training runs, so that we can
  # compare them.
  tb_viz = tb_op(
    log_dir_uri='%s/%s' % (working_dir, dsl.RUN_ID_PLACEHOLDER)
  )

  with dsl.ParallelFor(num_best_hps_list) as idx:  # start the full training runs in parallel

    train = train_op(
      data_dir=data_dir,
      workdir='%s/%s' % (tb_viz.outputs['log_dir_uri'], idx),
      tb_dir=tb_viz.outputs['log_dir_uri'],
      epochs=train_epochs, steps_per_epoch=steps_per_epoch,
      hp_idx=idx, hptune_results=hptune.outputs['hps']
      )

    eval_metrics = eval_metrics_op(
      thresholds=thresholds,
      metrics=train.outputs['metrics_output_path'],
      )

    with dsl.Condition(eval_metrics.outputs['deploy'] == 'deploy'):  # conditional serving
      serve = serve_op(
        model_path=train.outputs['train_output_path'],
        model_name='bikesw',
        namespace='default'
        )

    train.set_gpu_limit(2)
```

If you want, you can compile and run this pipeline the same way as was done in the previous section. You can also find this pipeline in the example repo [here](https://github.com/amygdala/code-snippets/blob/master/ml/kubeflow-pipelines/keras_tuner/example_pipelines/bw_ktune_metrics.py).

## More detail on the code, and requesting predictions from your model

This example didn't focus on the details of the pipeline component (step) implementations.  The training component uses a Keras model (TF 2.3). The serving component uses [TF-serving](https://www.tensorflow.org/tfx/guide/serving): once the serving service is up and running, you can send prediction requests to your trained model.

You can find more detail on these components, and an example of sending a prediction request, [here](https://github.com/amygdala/code-snippets/tree/master/ml/kubeflow-pipelines/keras_tuner).
