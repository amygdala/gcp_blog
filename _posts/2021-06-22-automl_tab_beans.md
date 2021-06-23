---
toc: true
layout: post
description: how you can use Vertex Pipelines to build an end-to-end ML workflow for training a custom model using AutoML
categories: [mlops, pipelines, vertex, kfp ]
title: Use Vertex Pipelines to build an AutoML Classification end-to-end workflow
---


## Introduction

This post shows how you can use [Vertex Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines) to build an end-to-end ML workflow that trains a custom model using AutoML; evaluates the accuracy of the trained model; and if the model is sufficiently accurate, deploys it to Vertex AI for serving.


### Vertex AI and Vertex Pipelines

The recently-launched [Vertex AI][1]  is a unified MLOps platform to help data scientists and ML engineers increase their rate of experimentation, deploy models faster, and manage models more effectively.  It brings AutoML and AI Platform together, in conjunction with some new MLOps-focused products, into a unified API, client library, and user interface.

[Vertex Pipelines][2] is part of [Vertex AI.][3]  It helps you to automate, monitor, and govern your ML systems by orchestrating your ML workflows.  It is automated, scalable, serverless, and cost-effective: you pay only for what you use.  Vertex Pipelines is the backbone of the Vertex AI ML Ops story, and makes it easy to build and run  ML workflows using any ML framework.  Because it is serverless, and has seamless integration with GCP and Vertex AI tools and services, you can focus on just building and running your pipelines without worrying about infrastructure or cluster maintenance.

Vertex Pipelines automatically logs metadata to track artifacts, lineage, metrics, and execution across your ML workflows, supports step execution caching, and provides support for enterprise security controls like [Cloud IAM][6], [VPC-SC][7], and [CMEK][8].

Vertex Pipelines supports two OSS Python SDKs: TFX ([TensorFlow Extended][4]) and KFP  ([Kubeflow Pipelines][5]).  The [example Vertex pipeline][9] highlighted in this post uses the KFP SDK, and includes use of the **[Google Cloud Pipeline Components][10]**, which support easy access to Vertex AI services. Vertex Pipelines requires v2 of the KFP SDK.   Soon, it will be possible to use the [KFP v2 ‘compatibility mode’][11] to run KFP V2 examples like this on OSS KFP as well.

## An end-to-end AutoML Workflow with Vertex Pipelines

[Vertex AI’s AutoML Tabular service][12] lets you bring your own structured data to train a model, without needing to build the model architecture yourself. For this example, I’ll use the UCI Machine Learning 'Dry beans dataset’[^1].  The task is a classification task: predict the type of a bean given some information about its characteristics.

Vertex Pipelines makes it very straightforward to construct a workflow to support building, evaluating, and deploying such models.   We’ll build a pipeline that looks like this:
<figure>
<a href="https://storage.googleapis.com/amy-jo/images/mp/beans.png" target="_blank"><img src="https://storage.googleapis.com/amy-jo/images/mp/beans.png" width="95%"/></a>
<figcaption><br/><i>The DAG for the AutoML classification workflow.</i></figcaption>
</figure>

You can see that the model deployment step is wrapped by a conditional: the model will only be deployed if the evaluation step indicates that it is sufficiently accurate.

For this example, nearly all the _components_ (steps) in the pipeline are prebuilt [Google Cloud Pipeline Components][14].  This means that we (mostly) just need to specify how the pipeline is put together using pre-existing building blocks.
However, I’ll add one Python function-based _custom component_ for model evaluation and metrics visualization.
The pipeline definition looks as follows (with a bit of detail elided):

```python
@kfp.dsl.pipeline(name="automl-tab-beans-training-v2",
                  pipeline_root=PIPELINE_ROOT)
def pipeline(
    bq_source: str = "bq://aju-dev-demos.beans.beans1",
    display_name: str = DISPLAY_NAME,
    project: str = PROJECT_ID,
    gcp_region: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
    thresholds_dict_str: str = '{"auRoc": 0.95}',
):
    dataset_create_op = gcc_aip.TabularDatasetCreateOp(
        project=project, display_name=display_name, bq_source=bq_source
    )

    training_op = gcc_aip.AutoMLTabularTrainingJobRunOp(
        project=project,
        display_name=display_name,
        optimization_prediction_type="classification",
        optimization_objective="minimize-log-loss",
        budget_milli_node_hours=1000,
        column_transformations=[
            {"numeric": {"column_name": "Area"}},
            {"numeric": {"column_name": "Perimeter"}},
            {"numeric": {"column_name": "MajorAxisLength"}},
            ... other columns ...
            {"categorical": {"column_name": "Class"}},
        ],
        dataset=dataset_create_op.outputs["dataset"],
        target_column="Class",
    )
    model_eval_task = classif_model_eval_metrics(
        project,
        gcp_region,
        api_endpoint,
        thresholds_dict_str,
        training_op.outputs["model"],
    )

    with dsl.Condition(
        model_eval_task.outputs["dep_decision"] == "true",
        name="deploy_decision",
    ):

        deploy_op = gcc_aip.ModelDeployOp(  # noqa: F841
            model=training_op.outputs["model"],
            project=project,
            machine_type="n1-standard-4",
        )
```

We first create a [_Dataset_][15] from a BigQuery table that holds the training data. Then, we use AutoML to train a tabular classification model.  The `dataset` arg to the training step gets its value from the output of the Dataset step (`dataset=dataset_create_op.outputs["dataset"]`).

After the model is trained, its evaluation metrics are checked against given ‘threshold’ information, to decide whether it’s accurate enough to deploy.
The next section goes into more detail about how this custom ‘eval metrics’ component is defined.  It takes as one of its inputs an output of the training step (`training_op.outputs["model"]`)— which points to the trained model.

Then, a KFP _conditional_ uses an output of the eval step to decide whether to proceed with the deployment:
```python
    with dsl.Condition(
        model_eval_task.outputs["dep_decision"] == "true",
        name="deploy_decision",
    ):
```
If the model is sufficiently accurate, the prebuilt deployment component is called.  This step creates an [_Endpoint_][16] and deploys the trained model to that endpoint for serving.

## Defining a custom component

Most of the steps in the pipeline above are drawn from pre-built components; building blocks that make it easy to construct an ML workflow.  But I’ve defined one custom component to parse the trained model’s evaluation metrics, render some metrics visualizations, and determine— based on given ‘threshold’ information— whether the model is good enough to be deployed.  This custom component is defined as a Python function with a `@kfp.v2.dsl.component` decorator.  When this function is evaluated, it is compiled to a task ‘factory function’ that can be used in a pipeline specification. The KFP SDK makes it very straightforward to define new pipeline components in this way.

Below is the custom component definition, with some detail elided.  The `@component` decorator specifies three optional args: the base container image to use; any packages to install; and the `yaml` file to which to write the component specification.

The component function, `classif_model_eval_metrics`, has some input parameters of note.  The `model` parameter is an input `kfp.v2.dsl.Model` artifact.  As you may remember from the pipeline specification above, here this input will be provided by an output of the training step.

The last two function args, `metrics` and `metricsc` , are component `Output`s, in this case of type `Metrics` and `ClassificationMetrics`.  They’re not explicitly passed as inputs to the component step, but rather are automatically instantiated and can be used in the component. E.g, in the function below, we’re calling `metricsc.log_roc_curve()` and `metricsc.log_confusion_matrix()` to render these visualizations in the Pipelines UI.  These `Output` params become component outputs when the component is compiled, and can be consumed by other pipeline steps.

The `NamedTuple` outputs are another type of component output.  Here we’re returning a string that indicates whether or not to deploy the model.

```python
@component(
    base_image="gcr.io/deeplearning-platform-release/tf2-cpu.2-3:latest",
    output_component_file="tables_eval_component.yaml",
    packages_to_install=["google-cloud-aiplatform"],
)
def classif_model_eval_metrics(
    project: str,
    location: str,  # "us-central1",
    api_endpoint: str,  # "us-central1-aiplatform.googleapis.com",
    thresholds_dict_str: str,
    model: Input[Model],
    metrics: Output[Metrics],
    metricsc: Output[ClassificationMetrics],
) -> NamedTuple("Outputs", [("dep_decision", str)]):  # Return parameter.

    import json
    import logging

    from google.cloud import aiplatform

    # Function to fetch model eval info
    def get_eval_info(client, model_name):
        ...

    # Use the given metrics threshold(s) to determine whether the model is
    # accurate enough to deploy.
    def classification_thresholds_check(metrics_dict, thresholds_dict):
        ...

    # Generate pipeline annotations and visualizations for the metrics info
    def log_metrics(metrics_list, metricsc):
		...
        metricsc.log_roc_curve(fpr, tpr, thresholds)
        ...
        metricsc.log_confusion_matrix(
            annotations,
            test_confusion_matrix["rows"],
        )

    aiplatform.init(project=project)
    # extract the model resource name from the input Model Artifact
    model_resource_path = model.uri.replace("aiplatform://v1/", "")
    logging.info("model path: %s", model_resource_path)

    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests to Vertex AI.
    client = aiplatform.gapic.ModelServiceClient(client_options=client_options)
    eval_name, metrics_list, metrics_str_list = get_eval_info(
        client, model_resource_path
    )

    log_metrics(metrics_list, metricsc)

    thresholds_dict = json.loads(thresholds_dict_str)
    deploy = classification_thresholds_check(metrics_list[0], thresholds_dict)
    if deploy:
        dep_decision = "true"
    else:
        dep_decision = "false"
    logging.info("deployment decision is %s", dep_decision)

    return (dep_decision,)
```

When this function is evaluated, we can use the generated factory function to define a pipeline step as part of a pipeline definition, as we saw in the previous section:

```python
model_eval_task = classif_model_eval_metrics(
    project,
    gcp_region,
    api_endpoint,
    thresholds_dict_str,
    training_op.outputs["model"],
)
```

The example notebook has the full component definition.

### Sharing component specifications

When the component is compiled, we can also indicate that a `yaml` component specification be generated.  We did this via the optional  `output_component_file="tables_eval_component.yaml"` arg passed to the `@component` decorator.
The `yaml` format allows the component specification to be put under version control and shared with others.

Then, the component can be used in other pipelines by calling the [`kfp.components.load_component_from_url` function][17] (and other variants like `load_component_from_file`).

## Running a pipeline job on Vertex Pipelines

Once a pipeline is defined, the next step is to _compile_ it — which generates a json job spec file— then submit and run it on Vertex Pipelines.  When you submit a pipeline job, you can specify values for pipeline input parameters, overriding their defaults.
The [example notebook][18] shows the details of how to do this.

Once a pipeline is running, you can view its details in the Cloud Console, including the pipeline run and lineage graphs shown above, as well as pipeline step logs and pipeline Artifact details.
You can also submit pipeline job specs via the Cloud Console UI, and the UI makes it easy to clone pipeline runs. The json pipeline specification file may also be put under version control and shared with others.


### Leveraging Pipeline step caching to develop and debug

Vertex Pipelines supports step caching, and this helps with iterating on pipeline development— when you rerun a pipeline, if a component’s inputs have not changed, its cached execution results can be reused. If you run this pipeline more than once, you might notice this feature in action.

If you’re playing along, try making a small change to the [example notebook][19] cell that holds the custom component definition (the `classif_model_eval_metrics` function in the “Define a metrics eval custom component” section) by uncommenting this line:
```python
  # metrics.metadata["model_type"] = "AutoML Tabular classification"
```

Then re-compile the component, recompile the pipeline **without changing the `DISPLAY_NAME` value**, and run it again. When you do so, you should see that Vertex Pipelines can leverage the cached executions for the upstream steps— as their inputs didn’t change— and only needs to re-execute from the changed component.  The pipeline DAG for the new run should look as follows, with the ‘recycle’ icon on some of the steps indicating that their cached executions were used.

<figure>
<a href="https://storage.googleapis.com/amy-jo/images/mp/beans_cached.png" target="_blank"><img src="https://storage.googleapis.com/amy-jo/images/mp/beans_cached.png" width="60%"/></a>
<figcaption><br/><i>Leveraging step caching with the AutoML classification workflow.</i></figcaption>
</figure>

> Note: Step caching is on by default, but if you want to disable it, you can pass the `enable_caching=False` arg to the `create_run_from_job_spec` function when you submit a pipeline run.

### Lineage tracking

If you click on an Artifact in a pipeline graph, you'll see a "VIEW LINEAGE" button.  This tracks how the artifacts are connected by step executions. So it’s kind of the inverse of the pipeline DAG, and can include multiple executions that consumed the same artifact (this can happen with cache hits, for example).
The tracking information shown is not necessarily  just for a single pipeline run, but for any pipeline execution that has used the given artifact.


<figure>
<a href="https://storage.googleapis.com/amy-jo/images/mp/beans_lineage_tracker.png" target="_blank"><img src="https://storage.googleapis.com/amy-jo/images/mp/beans_lineage_tracker.png" width="60%"/></a>
<figcaption><br/><i>Lineage tracking.</i></figcaption>
</figure>

## What’s next?

This post introduced Vertex Pipelines, and the prebuilt [Google Cloud Pipeline Components][20], which allow easy access to Vertex AI services.   The Pipelines example in this post uses the AutoML Tabular service, showing how straightforward it is to bring your own data to train a model. It showed a pipeline that creates a Dataset, trains a model using that dataset, obtains the model’s evaluation metrics, and decides whether or not to deploy the model to Vertex AI for serving.

For next steps, check out [other Vertex Pipelines example notebooks][21] as well as a [codelab][22] based in part on the pipeline in this post.
You  can also find other Vertex AI notebook examples [here][23] and [here][24].




[^1]:	from: KOKLU, M. and OZKAN, I.A., (2020) "Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques."In Computers and Electronics in Agriculture, 174, 105507. [DOI][13]

[1]:	https://cloud.google.com/vertex-ai/
[2]:	https://cloud.google.com/vertex-ai/docs/pipelines
[3]:	https://cloud.google.com/vertex-ai/
[4]:	https://www.tensorflow.org/tfx
[5]:	https://www.kubeflow.org/docs/components/pipelines/
[6]:	https://cloud.google.com/iam
[7]:	https://cloud.google.com/vpc-service-controls
[8]:	https://cloud.google.com/kms/docs/cmek
[9]:	https://github.com/GoogleCloudPlatform/ai-platform-samples/blob/master/ai-platform-unified/notebooks/official/pipelines/automl_tabular_classification_beans.ipynb
[10]:	https://github.com/kubeflow/pipelines/tree/master/components/google-cloud
[11]:	https://www.kubeflow.org/docs/components/pipelines/sdk/v2/v2-compatibility/#current-caveats
[12]:	https://cloud.google.com/vertex-ai/docs/start/automl-model-types#tabular
[13]:	https://doi.org/10.1016/j.compag.2020.105507
[14]:	https://github.com/kubeflow/pipelines/tree/master/components/google-cloud
[15]:	https://cloud.google.com/vertex-ai/docs/datasets/create-dataset-console
[16]:	https://cloud.google.com/vertex-ai/docs/general/deployment
[17]:	https://www.kubeflow.org/docs/components/pipelines/sdk/component-development/#using-your-component-in-a-pipeline
[18]:	https://github.com/GoogleCloudPlatform/ai-platform-samples/blob/master/ai-platform-unified/notebooks/official/pipelines/automl_tabular_classification_beans.ipynb
[19]:	https://github.com/GoogleCloudPlatform/ai-platform-samples/blob/master/ai-platform-unified/notebooks/official/pipelines/automl_tabular_classification_beans.ipynb
[20]:	https://github.com/kubeflow/pipelines/tree/master/components/google-cloud
[21]:	https://github.com/GoogleCloudPlatform/ai-platform-samples/tree/master/ai-platform-unified/notebooks/official/pipelines
[22]:	https://codelabs.developers.google.com/vertex-pipelines-intro?hl=en&continue=https%3A%2F%2Fcodelabs.developers.google.com%2F%3Fcat%3Dcloud#0
[23]:	https://github.com/GoogleCloudPlatform/ai-platform-samples/tree/master/ai-platform-unified/notebooks/official
[24]:	https://github.com/GoogleCloudPlatform/ai-platform-samples/tree/master/ai-platform-unified/notebooks/unofficial

