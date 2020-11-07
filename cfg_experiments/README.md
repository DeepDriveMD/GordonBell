# Parameters

| Parameter | Type    | Description                                      |
| --------- | ------- | ------------------------------------------------ |
| `md_counts` | int | Number of NAMD simulations to run concurrently each MD stage.|
| `ml_counts` | int | Number of ML models to train each ML stage. Should be the same as the number of hyperparameter configurations in `ml_hpo`|
| `node_counts` | int | Total number of compute nodes for NAMD simulations.|
| `gpu_per_node` | int | Number of GPUs available per compute node.|
| `cpu_per_node` | int | Number of CPUs available per compute node.|
| `cutoff` | int | Number of Angstroms defining closeness cutoff for creating contact maps.|
| `selection` | str | MDAnalysis selection string for preprocessing trajectories. Atom selection used to create point cloud and contact maps.|
| `epoch` | int | Number of training epochs per ML stage.|
| `sample_interval` | int | Selects every sample_interval'th point in the validation data set to save embeddings for each epoch.|
| `batch_size` | int | Inference mode batch size during outlier detection forward pass. Should be as large as possible to speed up computation while still fitting into memory.|
| `model_type` | str | Either `aae` to specify Adversarial Autoencoder or `vae` to specify variational autoencoder during inference.|
| `conda_pytorch` | str | Path to conda environment used in the ML and outlier detection stages.|
| `base_path` | str | Path to DeepDriveMD experiment output directory. Files from each stage will be written to this Directory.|
