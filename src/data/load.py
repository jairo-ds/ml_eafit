import torch
from torch.utils.data import TensorDataset
import argparse
import wandb
import os

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def load_datasetvolador(data_path, train_size=.8):
    """
    Load the datasetvolador data from local files.
    """
    # Cargar los datos desde archivos locales
    x_data = torch.load(os.path.join(data_path, 'x_data.pt'))
    y_data = torch.load(os.path.join(data_path, 'y_data.pt'))

    # Divide los datos en entrenamiento y validación
    train_size = int(len(x_data) * train_size)
    x_train, x_val = x_data[:train_size], x_data[train_size:]
    y_train, y_val = y_data[:train_size], y_data[train_size:]

    training_set = TensorDataset(x_train, y_train)
    validation_set = TensorDataset(x_val, y_val)
    test_set = TensorDataset(x_data, y_data)  # Ajusta esto si tienes un conjunto de prueba separado

    datasets = [training_set, validation_set, test_set]
    return datasets

def load_and_log(data_path):
    with wandb.init(
        project="MLOps-ds2024",
        name=f"Load Raw Data ExecId-{args.IdExecution}", job_type="load-data") as run:
        
        datasets = load_datasetvolador(data_path)  # Usar la nueva función para cargar datasetvolador
        names = ["training", "validation", "test"]

        raw_data = wandb.Artifact(
            "datasetvolador-raw", type="dataset",
            description="raw datasetvolador dataset, split into train/val/test",
            metadata={"source": "local files",
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        run.log_artifact(raw_data)

# Ruta a los datos en tu PC
data_path = './data/datasetvolador'

# testing
load_and_log(data_path)

