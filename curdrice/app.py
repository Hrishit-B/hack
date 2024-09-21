import os
import click
from curdrice.test import *
import textwrap

INVALID_FILETYPE_MSG = "Error: Invalid file format. %s must be a .csv file."
INVALID_PATH_MSG = "Error: Invalid file path/name. Path %s does not exist."

def validate_file(ctx, param, value):
    if not os.path.exists(value):
        raise click.BadParameter(INVALID_PATH_MSG % value)
    if not value.endswith('.csv'):
        raise click.BadParameter(INVALID_FILETYPE_MSG % value)
    return value

@click.group()
@click.option('-p', '--path', type=click.Path(exists=True), callback=validate_file, help="Loads and reads the dataset")
@click.option('-d', '--dest', type=click.Path(), help="Stores the zipped joblib files in the specified location")
@click.pass_context
def cli(ctx, path, dest):
    ctx.ensure_object(dict)
    ctx.obj['path'] = path
    ctx.obj['dest'] = dest

@cli.command()
@click.argument('target')
@click.pass_context
def classify(ctx, target):
    """Performs classification task on the dataset"""
    input_path = ctx.obj['path']
    output_path = ctx.obj['dest']

    print("dataset loaded!")

    files = ["LogReg.joblib", "NB.joblib", "SVC.joblib", "DTC.joblib", "RFC.joblib", "GBC.joblib", "Metrics.csv"]
    zip_file_name = output_path

    classification_testing(input_path, target, files, zip_file_name)
    for file in files:
        file_path = file
        if file_path != "Metrics.csv" and os.path.exists(file_path):
            os.remove(file_path)

    print("Classification models ran successfully!")

@cli.command()
@click.argument('feature_matrix')
@click.argument('target_vector')
@click.pass_context
def select(ctx, feature_matrix, target_vector):
    """Performs feature selection task on the dataset"""
    # input_path = ctx.obj['path']
    # output_path = ctx.obj['dest']
    # fs = automationizer.FeatureSelection(input_path, feature_matrix, target_vector, output_path)
    print("Feature Selection models ran successfully!")

@cli.command()
@click.argument('target')
@click.pass_context
def regress(ctx, target):
    """Performs regression task on the dataset"""
    input_path = ctx.obj['path']
    output_path = ctx.obj['dest']

    print("dataset loaded!")

    files = ["linear.joblib", "lasso.joblib", "decision_tree.joblib", "random_forest.joblib", "gradient_boosting.joblib"]
    zip_file_name = output_path

    regression_testing(input_path, target, files, zip_file_name)
    for file in files:
        file_path = file
        if os.path.exists(file_path):
            os.remove(file_path)

    print("Regression models ran successfully!")

@cli.command()
def compute():
    """Shows the computational resources needed to train the model"""
    print('saving computation efficiency graphs...')

@cli.command()
def analyze():
    """Compares the model performances across metrics"""
    pass

def curdrice():
    program_description = textwrap.dedent('''

  _______  __    __   ______    ___ _$$ |  ______  $$/   _______   ______  
 /       |/  |  /  | /      \  /    $$ | /      \ /  | /       | /      \ 
/$$$$$$$/ $$ |  $$ |/$$$$$$  |/$$$$$$$ |/$$$$$$  |$$ |/$$$$$$$/ /$$$$$$  |
$$ |      $$ |  $$ |$$ |  $$/ $$ |  $$ |$$ |  $$/ $$ |$$ |      $$    $$ |
$$ \_____ $$ \__$$ |$$ |      $$ \__$$ |$$ |      $$ |$$ \_____ $$$$$$$$/ 
$$       |$$    $$/ $$ |      $$    $$ |$$ |      $$ |$$       |$$       |
 $$$$$$$/  $$$$$$/  $$/        $$$$$$$/ $$/       $$/  $$$$$$$/  $$$$$$$/   

        Team Name : machinenotlearning

        Project Idea :

        Automation of the complete machine learning workflow: \n
        The main goal is to simplify the machine learning workflow for users with different levels of 
        expertise, through a CLI app. We ask the user for certain inputs like the dataset file or folder, 
        and the type of ML problem and target variable (if necessary). We then perform standard data preprocessing (not dataset specific) 
        and feature selection (if necessary), and train relevant models for the data and present a comparative analysis 
        of the models trained, along with downloadable model weight files in the joblib format.
    ''')
    
    cli.help = program_description
    cli()