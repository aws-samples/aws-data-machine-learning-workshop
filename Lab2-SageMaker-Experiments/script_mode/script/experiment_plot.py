## define a function to plot comparison of metrics between hyperparameters

# extract the table 
import matplotlib.pyplot as plt
import seaborn as sns
from sagemaker.analytics import ExperimentAnalytics

def extract_table(experiment_name,output_metric, parameter_name,hue_name):
    ## output_metrics: str of metric to be traced
    ## parameter_names: str of parameter to be analyzed
    
    trial_component_analytics = ExperimentAnalytics(experiment_name=experiment_name,parameter_names=[parameter_name]+[hue_name])
    analytic_table = trial_component_analytics.dataframe()
    
    #filter the dataframe further
    df_cols=list(["TrialComponentName"])+[output_metric]+[parameter_name]+[hue_name]
    sub_table=analytic_table[df_cols]
    
    plot_table=sub_table[sub_table[output_metric]>=0]

    #remove experiment string from the TrialComponentName to make it neater
    plot_table["run"]=plot_table["TrialComponentName"].str.replace(experiment_name+'-experiment-', '')
    
    return plot_table


def plot_table(table,experiment_name,output_metric, parameter_name, hue_name):
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    fig.suptitle('Experiment Analysis')

    # scatter plot here
    sns.scatterplot(ax=axes[0], x=table[parameter_name],y=table[output_metric], hue=table["run"])
    sns.scatterplot(ax=axes[1], x=table[parameter_name],y=table[output_metric], hue=table[hue_name])
    

    plt.tight_layout()
    plt.legend( title=experiment_name)
    plt.show()   
    return 

def analyze_experiment(experiment_name="training-job-experiment-1672305742-bcbe",output_metric="Test:loss - Last", parameter_name="hidden_channels", hue_name="optimizer"):
    
    print(experiment_name)   
    table=extract_table(experiment_name,output_metric, parameter_name,hue_name)
    print(experiment_name)
    plot_table(table,experiment_name,output_metric, parameter_name, hue_name )
    
    return table
