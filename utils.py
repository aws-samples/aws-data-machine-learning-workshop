import os
import tarfile
import pandas as pd
from pathlib import Path
import boto3
import time
import json
import datetime
import tqdm

region = boto3.Session().region_name
sm_client = boto3.client("sagemaker", region)
cw = boto3.Session().client("cloudwatch")
sm_runtime = boto3.client("sagemaker-runtime")

def create_tar(tarfile_name: str, local_path: Path):
    """
    Create a tar.gz archive with the content of `local_path`.
    """
    file_list = [k for k in local_path.glob("**/*.*") if f"{k.relative_to(local_path)}"[0] != "."]
    pbar = tqdm.tqdm(file_list, unit="files")
    with tarfile.open(tarfile_name, mode="w:gz") as archive:
        for k in pbar:
            pbar.set_description(f"{k}")
            archive.add(k, arcname=f"{k.relative_to(local_path)}")
    tar_size = Path(tarfile_name).stat().st_size / 10**6
    return tar_size


def list_model_metadata_df():
    """
    List the domain, framework, task, and model name of standard machine learning models found in 
    common model zoos.
    """
    list_model_metadata_response = sm_client.list_model_metadata()

    domains = []
    frameworks = []
    framework_versions = []
    tasks = []
    models = []

    for model_summary in list_model_metadata_response["ModelMetadataSummaries"]:
        domains.append(model_summary["Domain"])
        tasks.append(model_summary["Task"])
        models.append(model_summary["Model"])
        frameworks.append(model_summary["Framework"])
        framework_versions.append(model_summary["FrameworkVersion"])

    data = {
        "Domain": domains,
        "Task": tasks,
        "Framework": frameworks,
        "FrameworkVersion": framework_versions,
        "Model": models,
    }

    df = pd.DataFrame(data)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.colheader_justify", "center")
    pd.set_option("display.precision", 3)
    return df

def endpoint_creation_wait(endpoint_name: str):
    """
    Waiting for the endpoint to finish creation
    """
    describe_endpoint_response = sm_client.describe_endpoint(EndpointName=endpoint_name)

    while (describe_endpoint_response["EndpointStatus"] == "Creating") or (describe_endpoint_response["EndpointStatus"] == "Updating"):
        describe_endpoint_response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        print(describe_endpoint_response["EndpointStatus"])
        time.sleep(15)

    return describe_endpoint_response

def get_invocation_metrics_for_endpoint_variant(endpoint_name, variant_name, start_time, end_time):
    metrics = cw.get_metric_statistics(
        Namespace="AWS/SageMaker",
        MetricName="Invocations",
        StartTime=start_time,
        EndTime=end_time,
        Period=60,
        Statistics=["Sum"],
        Dimensions=[
            {"Name": "EndpointName", "Value": endpoint_name},
            {"Name": "VariantName", "Value": variant_name},
        ],
    )
    return (
        pd.DataFrame(metrics["Datapoints"])
        .sort_values("Timestamp")
        .set_index("Timestamp")
        .drop("Unit", axis=1)
        .rename(columns={"Sum": variant_name})
    )


def plot_endpoint_metrics(endpoint_name=None, variant1=None, variant2=None, start_time=None):
    start_time = start_time or datetime.now() - datetime.timedelta(minutes=60)
    end_time = datetime.datetime.now()
    metrics_variant1 = get_invocation_metrics_for_endpoint_variant(
        endpoint_name, variant1["VariantName"], start_time, end_time
    )
    metrics_variant2 = get_invocation_metrics_for_endpoint_variant(
        endpoint_name, variant2["VariantName"], start_time, end_time
    )
    metrics_variants = metrics_variant1.join(metrics_variant2, how="outer")
    metrics_variants.plot()
    return metrics_variants

def invoke_with_single_sentence(list_data, endpoint_name, variant_name):
    print(f"Sending test traffic to the endpoint {endpoint_name}. \nPlease wait...")
    predictions = []
    for payload in list_data:
        print(".", end="", flush=True)
        response = sm_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
            TargetVariant=variant_name,
        )
        predictions.append(response["Body"].read().decode("utf-8"))
        time.sleep(0.5)
    print('\nDone!')
    return predictions


def invoke_endpoint_for_two_minutes(endpoint_name=None):
    with open("../sample_payload/batch_data.csv", "r") as f:
        for row in f:
            print(".", end="", flush=True)
            payload = row.rstrip("\n")
            response = sm_runtime.invoke_endpoint(
                EndpointName=endpoint_name, ContentType="text/csv", Body=payload
            )
            response["Body"].read().decode("utf-8")
            time.sleep(1)
            
    return

def endpoint_update_wait(endpoint_name: str):
    """
    Waiting for the endpoint to finish update
    """
    while True:
        status = sm_client.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
        if status in ["InService", "Failed"]:
            print("Done")
            break
        print(".", end="", flush=True)
        time.sleep(5)

    return 

def invoke_endpoint_max_invocations(
    endpoint_name, max_invocations=100, wait_interval_sec=1, should_raise_exp=False
):
    print(f"Sending test traffic to the endpoint {endpoint_name}. \nPlease wait...")

    nm_rounds = round(max_invocations/100)
    count = 0
    for i in range(nm_rounds):
        with open("../sample_payload/batch_data.csv", "r") as f:
            for row in f:
                payload = row.rstrip("\n")
                try:
                    response = sm_runtime.invoke_endpoint(
                        EndpointName=endpoint_name, ContentType="text/csv", Body=payload
                    )
                    response["Body"].read()
                    print(".", end="", flush=True)
                except Exception as e:
                    print("E", end="", flush=True)
                    if should_raise_exp:
                        raise e
                count += 1
                if count > max_invocations:
                    break
                time.sleep(wait_interval_sec)

    print("\nDone!")
    return


def get_sagemaker_metrics(
    endpoint_name,
    endpoint_config_name,
    variant_name,
    metric_name,
    statistic,
    start_time,
    end_time,
):
    dimensions = [
        {"Name": "EndpointName", "Value": endpoint_name},
        {"Name": "VariantName", "Value": variant_name},
    ]
    if endpoint_config_name is not None:
        dimensions.append({"Name": "EndpointConfigName", "Value": endpoint_config_name})
    metrics = cw.get_metric_statistics(
        Namespace="AWS/SageMaker",
        MetricName=metric_name,
        StartTime=start_time,
        EndTime=end_time,
        Period=60,
        Statistics=[statistic],
        Dimensions=dimensions,
    )
    rename = endpoint_config_name if endpoint_config_name is not None else "ALL"
    if len(metrics["Datapoints"]) == 0:
        return
    return (
        pd.DataFrame(metrics["Datapoints"])
        .sort_values("Timestamp")
        .set_index("Timestamp")
        .drop(["Unit"], axis=1)
        .rename(columns={statistic: rename})
    )


def plot_endpoint_invocation_metrics(
    endpoint_name,
    endpoint_config_name,
    variant_name,
    metric_name,
    statistic,
    start_time=None,
):
    start_time = start_time or datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=60)
    end_time = datetime.datetime.now(datetime.timezone.utc)
    metrics_variants = get_sagemaker_metrics(
        endpoint_name,
        endpoint_config_name,
        variant_name,
        metric_name,
        statistic,
        start_time,
        end_time,
    )
    if metrics_variants is None:
        return
    metrics_variants.plot(title=f"{metric_name}-{statistic}")
    return metrics_variants

def create_auto_rollback_alarm(
    alarm_name, endpoint_name, variant_name, metric_name, statistic, threshold
):
    cw.put_metric_alarm(
        AlarmName=alarm_name,
        AlarmDescription="Test SageMaker endpoint deployment auto-rollback alarm",
        ActionsEnabled=False,
        Namespace="AWS/SageMaker",
        MetricName=metric_name,
        Statistic=statistic,
        Dimensions=[
            {"Name": "EndpointName", "Value": endpoint_name},
            {"Name": "VariantName", "Value": variant_name},
        ],
        Period=60,
        EvaluationPeriods=1,
        Threshold=threshold,
        ComparisonOperator="GreaterThanOrEqualToThreshold",
        TreatMissingData="notBreaching",
    )