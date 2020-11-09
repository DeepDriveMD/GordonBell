# Parameters

| Parameter | Type    | Description                                      |
| --------- | ------- | ------------------------------------------------ |
| `md_counts` | int | Number of NAMD simulations to run concurrently each MD stage.|
| `ml_counts` | int | Number of ML models to train each ML stage. Should be the same as the number of hyperparameter configurations in `ml_hpo`.|
| `node_counts` | int | Total number of compute nodes for NAMD simulations.|
| `gpu_per_node` | int | Number of GPUs available per compute node.|
| `cpu_per_node` | int | Number of CPUs available per compute node.|
| `cutoff` | int | Number of Angstroms defining closeness cutoff for creating contact maps.|
| `selection` | str | MDAnalysis selection string for preprocessing trajectories. Atom selection used to create point cloud and contact maps.|
| `epoch` | int | Number of training epochs per ML stage.|
| `sample_interval` | int | Selects every sample_interval'th point in the validation data set to save embeddings for each epoch.|
| `batch_size` | int | Inference mode batch size during outlier detection forward pass. Should be as large as possible to speed up computation while still fitting into memory.|
| `model_type` | str | Either `aae` to specify adversarial autoencoder or `vae` to specify variational autoencoder during inference.|
| `conda_pytorch` | str | Path to conda environment used in the ML and outlier detection stages.|
| `base_path` | str | Path to DeepDriveMD experiment output directory. Files from each stage will be written to this Directory.|
| `molecules_path` | str | Path to [molecules](https://github.com/braceal/molecules/tree/gb-2020) repository clone where deep learning models are implemented.|
| `namd_path` | str | Path to NAMD executable.|
| `namd_log_filename` | str | Name of NAMD log file.|
| `system_name` | str | This will name a subdirectory inside `base_path` to write DeepDriveMD results.|
| `residues` | int | Number of residues in the protein atom selection.|
| `dcdfreq` | int | NAMD parameter: timesteps between writing coordinates to trajectory file.|
| `num_steps_min` | int | Number of timesteps to run energy minimization.|
| `num_steps_eq` | int | Number of timesteps to run NAMD simulations.|
| `stepspercycle` | int | NAMD parameter: timesteps per cycle.|
| `fullElectFrequency` | int | NAMD parameter: number of timesteps between full electrostatic evaluations.|
| `margin` | int | NAMD parameter: extra length in patch dimension (Å). Special note: if this optional parameter appears in the configuration it will set the margin to 3 Å indepedent of the value passed. If margin is not specified then the simulation will not use any margin.|
| `init_weights` | str | Path to pretrained model weights, must have .pt extension. Note: model dimensions must be compatable with `residues`.|
| `ref_pdb` | str | Path to PDB file used as a reference for computing RMSD during preprocessing.|
| `pdb_file` | list | List of PDB file paths to use as starting spawns for NAMD simulations. If `md_counts` is greater than then number of PDBs passed, simulations will reuse PDB files in a cyclical order.|
| `ml_hpo` | list | List of dictionaries specifying model hyperparameters for ML stage. DeepDriveMD expects `ml_counts` hyperparameter dictionaries.|
| `ml_hpo.optimizer` | str | Name of PyTorch optimizer and learning rate to use for training ML models. Names follow [PyTorch naming convention](https://pytorch.org/docs/stable/optim.html).|
| `ml_hpo.loss_weights` | str | Hyperparameters specifying how much relative weight to put on the reconstruction and gradient penalty in the [adversarial autoencoder](https://github.com/braceal/molecules/blob/gb-2020/examples/example_aae.py) model.|
| `ml_hpo.latent_dim` | int | Latent dimension of model.|
| `ml_hpo.batch_size` | int | Batch size used for model training.|
| `CUR_STAGE` | int | Starting iteration of DeepDriveMD, should be 0.|
| `MAX_STAGE` | int | Total number of iterations to run DeepDriveMD.|
| `RETRAIN_FREQ` | int | How often to retrain ML model, should be 1.|
| `resource` | str | radical.entk res_dict parameter: HPC resource e.g. 'ornl.summit'|
| `queue` | str | radical.entk res_dict parameter: 'batch' or 'killable'|
| `schema` | str | radical.entk res_dict parameter: 'local'|
| `walltime` | int | Number of minutes to run job for.|
| `project` | int | HPC project id.|
