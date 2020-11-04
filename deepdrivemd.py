import os
import pathlib
import sys
import json
import time
import itertools

import radical.utils as ru

from radical.entk import Pipeline, Stage, Task, AppManager

from namd_config import write_namd_configuration
#

def generate_training_pipeline(cfg):
    """
    Function to generate the CVAE_MD pipeline
    """
    CUR_STAGE = cfg["CUR_STAGE"]
    MAX_STAGE = cfg["MAX_STAGE"]

    def generate_MD_stage(num_MD=1):
        """
        Function to generate MD stage.
        """
        s1 = Stage()
        s1.name = "MD"

        initial_MD = True
        outlier_filepath = "%s/Outlier_search/restart_points.json" % cfg["base_path"]

        if os.path.exists(outlier_filepath):
            initial_MD = False
            with open(outlier_filepath, "r") as f:
                outlier_list = json.load(f)
        else:
            outlier_list = itertools.cycle(cfg['pdb_file'])

        # MD tasks
        time_stamp = int(time.time())
        for i in range(num_MD):
            t1 = Task()

            t1.pre_exec += [
                "module unload prrte",
                "module load cuda",
                "module load spectrum-mpi",
                "module load fftw",
                "export LD_LIBRARY_PATH=/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/xl-16.1.1-5/spectrum-mpi-10.3.1.2-20200121-p6nrnt6vtvkn356wqg6f74n6jspnpjd2/lib/pami_port:$LD_LIBRARY_PATH",
                "export LD_PRELOAD=/opt/ibm/spectrum_mpi/lib/libpami_cudahook.so:$LD_PRELOAD",
                "unset CUDA_VISIBLE_DEVICES",
            ]

            t1.pre_exec += ["mkdir -p %s/MD_exps/%s" % (cfg["base_path"], cfg["system_name"]),
                            "cd %s/MD_exps/%s" % (cfg["base_path"], cfg["system_name"])]
            t1.pre_exec += [
                "mkdir -p omm_runs_%d && cd omm_runs_%d"
                % (time_stamp + i, time_stamp + i)
            ]

            cmd_cat = "cat /dev/null"
            cmd_jsrun = "jsrun --bind rs -n%s -p%s -r%s -g%s -c%s" % (
                cfg["gpu_per_node"] * cfg["node_counts"],
                cfg["gpu_per_node"] * cfg["node_counts"],
                cfg["gpu_per_node"],
                1,
                cfg["cpu_per_node"] // cfg["gpu_per_node"],
            )
            cmd_namd = cfg["namd_path"]
            t1.executable = ["%s; %s %s" % (cmd_cat, cmd_jsrun, cmd_namd)]

            omm_dir = "%s/MD_exps/%s/omm_runs_%d" % (
                cfg["base_path"],
                cfg["system_name"],
                time_stamp + i,
            )

            # pick initial point of simulation
            if initial_MD:
                pdb_path = next(outlier_list)
            else:
                pdb_path = outlier_list[i]

            conf_path = os.path.join("%s/tmp/%s.conf" % (cfg["base_path"], time_stamp + i))
            write_namd_configuration(conf_path, pdb_path,
                    str(cfg['dcdfreq']),str(cfg['num_steps_eq']))
            t1.pre_exec += [
                "cp %s %s" % (pdb_path, omm_dir),
                "cp %s %s" % (conf_path, omm_dir),
            ]
            t1.arguments = [
                "+ignoresharing",
                "+ppn",
                "7",
                "+pemap",
                "0-83:4,88-171:4",
                conf_path,
            ]

            t1.download_output_data = ["STDOUT > %s_%s" % (i, cfg["namd_log_filename"])]

            # assign hardware the task
            t1.cpu_reqs = {
                "processes": 6 * cfg["node_counts"],
                "process_type": "MPI",
                "threads_per_process": 6 * 4,
                "thread_type": "OpenMP",
            }
            t1.gpu_reqs = {
                "processes": 1,
                "process_type": None,
                "threads_per_process": 1,
                "thread_type": "CUDA",
            }

            # Add the MD task to the simulating stage
            s1.add_tasks(t1)
        return s1

    def generate_aggregating_stage():
        """
        Function to concatenate the MD trajectory (h5 contact map)
        """
        s2 = Stage()
        s2.name = "aggregating"

        # Aggregation task
        t2 = Task()

        t2.pre_exec = [
            ". /sw/summit/python/3.6/anaconda3/5.3.0/etc/profile.d/conda.sh || true",
            "conda activate %s" % cfg["conda_pytorch"],
            "export LANG=en_US.utf-8",
            "export LC_ALL=en_US.utf-8",
        ]
        # preprocessing for molecules' script, it needs files in a single
        # directory
        # the following pre-processing does:
        # 1) find all (.dcd) files from openmm results
        # 2) create a temp directory
        # 3) symlink them in the temp directory
        t2.pre_exec += [
            "export dcd_list=(`ls %s/MD_exps/%s/omm_runs_*/*dcd`)"
            % (cfg["base_path"], cfg["system_name"]),
            "export tmp_path=`mktemp -p %s/MD_to_CVAE/ -d`" % cfg["base_path"],
            "for dcd in ${dcd_list[@]}; do tmp=$(basename $(dirname $dcd)); ln -s $dcd $tmp_path/$tmp.dcd; done",
            "ln -s %s $tmp_path/prot.pdb" % cfg["ref_pdb"],
            "ls ${tmp_path}",
        ]

        t2.pre_exec += ["unset CUDA_VISIBLE_DEVICES", "export OMP_NUM_THREADS=4"]

        # - Each node takes 6 ranks
        # - each rank processes 2 files
        # - each iteration accumulates files to process
        cnt_constraint = min(
            cfg["node_counts"] * 6, cfg["md_counts"] * max(1, CUR_STAGE) // 2
        )

        t2.executable = ["%s/bin/python" % (cfg["conda_pytorch"])]  # MD_to_CVAE.py
        t2.arguments = [
            "%s/scripts/traj_to_dset.py" % cfg["molecules_path"],
            "-t", "$tmp_path",
            "-p", cfg["ref_pdb"],
            "-r", cfg["ref_pdb"],
            "-o", "%s/MD_to_CVAE/cvae_input.h5" % cfg["base_path"],
            "--contact_maps_parameters", "kernel_type=threshold,threshold=%s" % cfg["cutoff"],
            "-s", cfg["selection"],
            "--rmsd",
            "--fnc",
            "--contact_map",
            "--point_cloud",
            "--num_workers", 2,
            "--distributed",
            "--verbose",
        ]

        # Add the aggregation task to the aggreagating stage
        t2.cpu_reqs = {
            "processes": 1 * cnt_constraint,
            "process_type": "MPI",
            "threads_per_process": 6 * 4,
            "thread_type": "OpenMP",
        }

        s2.add_tasks(t2)
        return s2

    def generate_ML_stage(num_ML=1):
        """
        Function to generate the learning stage
        """
        # learn task
        time_stamp = int(time.time())
        stages = []
        for i in range(num_ML):
            s3 = Stage()
            s3.name = "learning"

            t3 = Task()
            t3.pre_exec = [
                ". /sw/summit/python/3.6/anaconda3/5.3.0/etc/profile.d/conda.sh || true"
            ]
            t3.pre_exec += [
                "module load gcc/7.4.0 || module load gcc/7.3.1",
                "module load cuda/10.1.243",
                "module load hdf5/1.10.4 || true",
                "export LANG=en_US.utf-8",
                "export LC_ALL=en_US.utf-8",
                "export HDF5_USE_FILE_LOCKING=FALSE"
            ]
            t3.pre_exec += ["conda activate %s" % cfg["conda_pytorch"]]
            dim = i + 3
            cvae_dir = "cvae_runs_%.2d_%d" % (dim, time_stamp + i)
            t3.pre_exec += ["cd %s/CVAE_exps" % cfg["base_path"]]
            t3.pre_exec += [
                "export LD_LIBRARY_PATH=/gpfs/alpine/proj-shared/med110/atrifan/scripts/cuda/targets/ppc64le-linux/lib/:$LD_LIBRARY_PATH"
            ]
            t3.pre_exec += [
                "export LD_LIBRARY_PATH=/usr/workspace/cv_ddmd/lee1078/anaconda/envs/cuda/targets/ppc64le-linux/lib/:$LD_LIBRARY_PATH"
            ]
            # t3.pre_exec += ['mkdir -p %s && cd %s' % (cvae_dir, cvae_dir)] # model_id creates sub-dir
            # this is for ddp, distributed
            t3.pre_exec += ["unset CUDA_VISIBLE_DEVICES", "export OMP_NUM_THREADS=4"]
            # pnodes = cfg['node_counts'] // num_ML # partition
            pnodes = 1  # max(1, pnodes)

            hp = cfg["ml_hpo"][i]
            cmd_cat = "cat /dev/null"
            cmd_jsrun = "jsrun -n %s -g %s -a %s -c %s -d packed" % (
                pnodes,
                cfg["gpu_per_node"],
                cfg["gpu_per_node"],
                cfg["cpu_per_node"],
            )

            # AAE config
            cmd_vae = "%s/examples/bin/run_aae_dist_entk.sh" % cfg["molecules_path"]

            t3.executable = ["%s; %s %s" % (cmd_cat, cmd_jsrun, cmd_vae)]
            t3.arguments = ["%s/bin/python" % cfg["conda_pytorch"]]
            t3.arguments += [
                "%s/examples/example_aae.py" % cfg["molecules_path"],
                "-i",  "%s/MD_to_CVAE/cvae_input.h5" % cfg["base_path"],
                "-o", "./",
                #"--distributed",
                "-m", cvae_dir,
                "-dn", "point_cloud",
                "-rn", "rmsd",
                "--encoder_kernel_sizes", 5, 3, 3, 1, 1,
                "-nf", 0,
                "-np", str(cfg["residues"]),
                "-e", str(cfg["epoch"]),
                "-b", str(hp["batch_size"]),
                "-opt", hp["optimizer"],
                "-iw", cfg["init_weights"],
                "-lw", hp["loss_weights"],
                "-S", str(cfg["sample_interval"]),
                "-ti", str(int(cfg["epoch"]) + 1),
                "-d", str(hp["latent_dim"]),
                "--num_data_workers", 0,
            ]

            t3.cpu_reqs = {
                "processes": 6,
                "process_type": "MPI",
                "threads_per_process": 4,
                "thread_type": "OpenMP",
            }
            t3.gpu_reqs = {
                "processes": 1,
                "process_type": "MPI",
                "threads_per_process": 1,
                "thread_type": "CUDA",
            }

            # Add the learn task to the learning stage
            s3.add_tasks(t3)
            stages.append(s3)
        return stages

    def generate_interfacing_stage():
        s4 = Stage()
        s4.name = "scanning"

        # Scaning for outliers and prepare the next stage of MDs
        t4 = Task()

        t4.pre_exec = [
            ". /sw/summit/python/3.6/anaconda3/5.3.0/etc/profile.d/conda.sh || true"
        ]
        t4.pre_exec += ["conda activate %s" % cfg["conda_pytorch"]]
        t4.pre_exec += ["mkdir -p %s/Outlier_search/outlier_pdbs" % cfg["base_path"]]
        t4.pre_exec += [
            'export models=""; for i in `ls -d %s/CVAE_exps/model-cvae_runs*/`; do if [ "$models" != "" ]; then    models=$models","$i; else models=$i; fi; done;cat /dev/null'
            % cfg["base_path"]
        ]
        t4.pre_exec += ["export LANG=en_US.utf-8", "export LC_ALL=en_US.utf-8"]
        t4.pre_exec += ["unset CUDA_VISIBLE_DEVICES", "export OMP_NUM_THREADS=4"]

        cmd_cat = "cat /dev/null"
        cmd_jsrun = "jsrun -n %s -a %s -g %s -r 1 -c %s" % (
            cfg["node_counts"],
            cfg["gpu_per_node"],
            cfg["gpu_per_node"],
            cfg["cpu_per_node"] // cfg["gpu_per_node"],
        )

        t4.executable = [
            " %s; %s %s/examples/outlier_detection/run_optics_dist_entk.sh"
            % (cmd_cat, cmd_jsrun, cfg["molecules_path"])
        ]
        t4.arguments = ["%s/bin/python" % cfg["conda_pytorch"]]
        t4.arguments += [
            "%s/examples/outlier_detection/optics.py" % cfg["molecules_path"],
            "--sim_path",  "%s/MD_exps/%s" % (cfg["base_path"], cfg["system_name"]),
            "--pdb_out_path",  "%s/Outlier_search/outlier_pdbs" % cfg["base_path"],
            "--restart_points_path", "%s/Outlier_search/restart_points.json" % cfg["base_path"],
            "--data_path", "%s/MD_to_CVAE/cvae_input.h5" % cfg["base_path"],
            "--model_paths", "$models",
            "--model_type",  cfg["model_type"],
            "--min_samples", 10,
            "--n_outliers", cfg['md_counts'] ,
            "--dim1", str(cfg["residues"]),
            "--dim2", str(cfg["residues"]),
            "--cm_format", "sparse-concat",
            "--batch_size", str(cfg["batch_size"]),
            "--distributed",
        ]

        t4.cpu_reqs = {
            "processes": 6 * cfg["node_counts"],
            "process_type": "MPI",
            "threads_per_process": 6 * 4,
            "thread_type": "OpenMP",
        }
        t4.gpu_reqs = {
            "processes": 1,
            "process_type": "MPI",
            "threads_per_process": 1,
            "thread_type": "CUDA",
        }

        s4.add_tasks(t4)
        s4.post_exec = func_condition
        return s4

    def func_condition():
        nonlocal CUR_STAGE
        nonlocal MAX_STAGE
        if CUR_STAGE < MAX_STAGE:
            func_on_true()
        else:
            func_on_false()

    def func_on_true():
        nonlocal CUR_STAGE
        nonlocal MAX_STAGE
        print("finishing stage %d of %d" % (CUR_STAGE, MAX_STAGE))

        # --------------------------
        # MD stage
        s1 = generate_MD_stage(num_MD=cfg["md_counts"])
        # Add simulating stage to the training pipeline
        p.add_stages(s1)

        # --------------------------
        # Aggregate stage
        s2 = generate_aggregating_stage()
        p.add_stages(s2)

        if CUR_STAGE % cfg["RETRAIN_FREQ"] == 0:
            # --------------------------
            # Learning stage
            s3 = generate_ML_stage(num_ML=cfg["ml_counts"])
            # Add the learning stage to the pipeline
            p.add_stages(s3)

        # --------------------------
        # Outlier identification stage
        s4 = generate_interfacing_stage()
        p.add_stages(s4)

        CUR_STAGE += 1

    def func_on_false():
        print("Done")

    p = Pipeline()
    p.name = "MD_ML"

    # --------------------------
    # MD stage
    s1 = generate_MD_stage(num_MD=cfg["md_counts"])
    # Add simulating stage to the training pipeline
    p.add_stages(s1)

    # --------------------------
    # Aggregate stage
    s2 = generate_aggregating_stage()
    # Add the aggregating stage to the training pipeline
    p.add_stages(s2)

    # --------------------------
    # Learning stage
    s3 = generate_ML_stage(num_ML=cfg["ml_counts"])
    # Add the learning stage to the pipeline
    p.add_stages(s3)

    # --------------------------
    # Outlier identification stage
    s4 = generate_interfacing_stage()
    p.add_stages(s4)

    CUR_STAGE += 1

    return p


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    reporter = ru.Reporter(name="radical.entk")
    reporter.title("COVID-19 - Workflow2")

    # resource specified as argument
    if len(sys.argv) == 2:
        cfg_file = sys.argv[1]
    elif sys.argv[0] == "molecules_adrp.py":
        cfg_file = "adrp_system.json"
    elif sys.argv[0] == "molecules_3clpro.py":
        cfg_file = "3clpro_system.json"
    else:
        reporter.exit("Usage:\t%s [config.json]\n\n" % sys.argv[0])

    cfg = ru.Config(cfg=ru.read_json(cfg_file))
    if "node_counts" not in cfg:
        cfg["node_counts"] = max(1, cfg["md_counts"] // cfg["gpu_per_node"])

    res_dict = {
        "resource": cfg["resource"],
        "queue": cfg["queue"],
        "schema": cfg["schema"],
        "walltime": cfg["walltime"],
        "project": cfg["project"],
        "cpus": 42 * 4 * cfg["node_counts"],
        "gpus": cfg["node_counts"] * cfg["gpu_per_node"],
    }

    # Create Application Manager
    appman = AppManager(
        hostname=os.environ.get("RMQ_HOSTNAME"),
        port=int(os.environ.get("RMQ_PORT")),
        username=os.environ.get("RMQ_USERNAME"),
        password=os.environ.get("RMQ_PASSWORD"),
    )
    appman.resource_desc = res_dict

    pathlib.Path("%s/tmp" % cfg["base_path"]).mkdir(exist_ok=True)
    p1 = generate_training_pipeline(cfg)
    pipelines = [p1]

    # Assign the workflow as a list of Pipelines to the Application Manager. In
    # this way, all the pipelines in the list will execute concurrently.
    appman.workflow = pipelines

    # Run the Application Manager
    appman.run()
