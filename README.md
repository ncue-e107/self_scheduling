# 🚀 PBT-Ray-Scheduler: Adaptive Resource Allocation for Population-Based Training

This project implements an advanced distributed **Population-Based Training (PBT)** framework using **Ray** for high-performance hyperparameter optimization. It features two novel adaptive scheduling strategies—**ERA** (Exponential Resource Allocation) and **ETA** (Execution Time-aware Allocation)—and includes a specialized **Garbage-Collecting/Pruning Allocation (GPA)** mechanism to dynamically retire underperforming nodes.

## 🌟 Key Features

* **Distributed PBT on Ray:** Seamlessly executes PyTorch-based deep learning trials across a distributed cluster.
* **Adaptive Scheduling Strategies:**
  * **ERA (Exponential Resource Allocation):** Dynamically assigns a decreasing number of trials to resources based on node performance scores and the current training stage.
  * **ETA (Execution Time-aware Allocation):** Dynamically assigns trials based on the estimated execution time of tasks on each node (slower nodes get fewer tasks).
* **Resource Pruning (GPA):** The `PBT_GPA.py` scheduler includes a **Garbage-collecting/Pruning** mechanism. It identifies and retires the weakest idle nodes when training reaches a "late stage," freeing up resources and focusing on high-performance nodes.
* **PBT Mutation:** Implements standard PBT logic with **Exploit** (copying top performers) and **Explore** (random hyperparameter perturbation).

## 📦 Project Structure

```text
.
├── PBT_SPA.py          # Standard PBT Scheduler (Supports ERA/ETA modes)
├── PBT_GPA.py          # PBT Scheduler with Node Pruning (Supports ERA/ETA modes)
├── score.py            # Utility script to benchmark nodes and generate score.json
├── score.json          # Node resource allocation and performance scores (Required by PBT scripts)
├── res.json            # (Optional) Initial resource config input for score.py
├── requirements.txt    # Python dependencies list
└── README.md
```

## 🛠️ Setup and Installation

#### 1. **Prerequisite: Conda Environment**

It assignss recommended to use **Conda** to manage the Python environment.

```bash
# Create a new conda environment (Python 3.10 recommended)
conda create -n pbt_ray python=3.10 -y

# Activate the environment
conda activate pbt_ray
```

#### 2. Install Dependencies

Install the required Python packages using the provided `requirements.txt`.
Note that this project requires **PyTorch** and **Ray** with **CUDA 12.x** support (as specified in the requirements).

#### 3. Configure Ray Cluster

Ensure your Ray cluster is active. You must update the `HEAD_NODE_IP` variable in `PBT_SPA.py`
, `PBT_GPA.py`, and `score.py` to match your cluster's actual Head Node IP.

```python
# Example in PBT_SPA.py
HEAD_NODE_IP = "xxx.xxx.xxx.xxx"
```

#### 4. Generate Node Scores (`score.json`)

The schedulers rely on a `score.json` file to understand the capability of each node.

  1. **Prepare `res.json`**:Edit `res.json` defining your available IPs if you want `score.py` to target specific nodes.

  ```
  {
    "CPU": {
        "xxx.xxx.xxx.xxx": EXCHANGE_YOUR_CPU_CORE_NUM
    },
    "GPU": {
        "xxx.xxx.xxx.xxx": 1
    }
  }

  ```

  2. **Run Benchmark**: Execute `score.py` on the cluster. It will run dummy training tasks to calculate the throughput (score) of each node.

## 🚀 How to execute

You can choose between the Standard Scheduler (`PBT_SPA.py`) or the Pruning Scheduler (`PBT_GPA.py`). Both support `ERA` and `ETA` modes.

#### Option 1: Standard PBT (`PBT_SPA.py`)

Use this for standard adaptive scheduling without retiring nodes.

```bash
# Run with Exponential Resource Allocation (ERA) - 5 experiments
python PBT_SPA.py --mode ERA --exp_times 5

# Run with Execution Time-aware Allocation (ETA) - 1 experiment
python PBT_SPA.py --mode ETA --exp_times 1
```

#### Option 2: PBT GPA mode (`PBT_GPA.py`)

Use this to enable the retiring mechanism. Weak nodes will be removed from the available pool during the late stages of training.

```bash
# Run ERA with Pruning
python PBT_GPA.py --mode ERA --exp_times 1

# Run ETA with Pruning
python PBT_GPA.py --mode ETA --exp_times 1
```

## 📈 Configuration Parameters

|Parameter|Default|Description|
|---|---|---|
|`HYPER_NUM`|`50`|Total population size (number of hyperparameter sets).|
|`STOP_ITER`|`1000`|Maximum iterations per trial.|
|`STOP_ACC`|`0.8`|Target accuracy for early stopping.|
|`INTERVAL_CHECK`|`50`|Frequency (in iterations) for PBT exploit/explore checks.|
|`BATCH_SIZE`|`[32, 64, 128, 256, 512]`|Discrete set of batch sizes for mutation.|
|`MAX_RETIRE_NODES`|`9`|(GPA Only) Max number of weak nodes to retire.|
|`LATE_STAGE`|`0.8`|(GPA Only) Percentage of STOP_ITER to trigger retiring.|

## 📊 Output Results

Experiment logs and results are saved in `log_ERA/` or `log_ETA/`:

* `Running_Results.txt`: Summary of experiment configurations, average runtime/accuracy, and historical batch size usage per node.
* `communication_time/*.txt`: Detailed logs of communication overhead vs. computation time.
* `results/.../[id]-accuracy.json`: Per-trial accuracy and iteration logs.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
```
MIT License

Copyright (c) 2025 [Your Name or Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
## 🖇️Acknowledgements

- [Python Ray](https://github.com/ray-project/ray)
- [PyTorch](https://github.com/pytorch/pytorch)
