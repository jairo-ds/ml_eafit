import torch
from torch.utils.data import TensorDataset
import os
import argparse
import wandb

# Definir argumentos del script
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

def preprocess(dataset, normalize=True, expand_dims=True):
    """
    Prepara los datos.
    """
    x, y = dataset.tensors

    if normalize:
        # Escala los datos al rango [0, 1]
        x = x.type(torch.float32) / 255

    if expand_dims:
        # Asegúrate de que los datos tienen la forma adecuada para el modelo (por ejemplo, añadiendo un canal si es necesario)
        x = torch.unsqueeze(x, 1)
    
    return TensorDataset(x, y)

def preprocess_and_log(steps):

    with wandb.init(project="Integradorpj-2024", name=f"Preprocess Data ExecId-{args.IdExecution}", job_type="preprocess-data") as run:    
        processed_data = wandb.Artifact(
            "datasetvolador-preprocess", type="dataset",
            description="Preprocessed datasetvolador dataset",
            metadata=steps)
         
        # Declarar el artifact que usaremos
        raw_data_artifact = run.use_artifact('datasetvolador-raw:latest')

        # Descargar el artifact si es necesario
        raw_dataset = raw_data_artifact.download(root="./data/artifacts/")
        
        for split in ["training", "validation", "test"]:
            raw_split = read(raw_dataset, split)
            processed_dataset = preprocess(raw_split, **steps)

            with processed_data.new_file(split + ".pt", mode="wb") as file:
                x, y = processed_dataset.tensors
                torch.save((x, y), file)

        run.log_artifact(processed_data)

def read(data_dir, split):
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))
    return TensorDataset(x, y)

# Definir los pasos de preprocesamiento
steps = {"normalize": True, "expand_dims": False}

# Ejecutar la función de preprocesamiento y registro
preprocess_and_log(steps)
