import time
import torch
import math
import torch.nn as nn
import torch.optim as optim
import ray
import socket
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
# from tqdm import tqdm # 註解掉 tqdm，在 Ray 遠程任務中可能導致日誌混亂
from pathlib import Path
import json

# 頭節點IP
HEAD_NODE_IP = "PLEASE EXCHANGE YOUR HEAD NODE IP!"

# 資源分配
RESOURCE_ALLOCATION = {}

# 確保 res.json 路徑正確
res_json_path = Path("./res.json")
if not res_json_path.exists():
    print(f"Error: res.json not found at {res_json_path.resolve()}")
    # 使用一個範例結構，避免程式崩潰
    RESOURCE_ALLOCATION = {
        "CPU": {"127.0.0.1": 4},
        "GPU": {}
    }
else:
    with res_json_path.open("r") as f:
        RESOURCE_ALLOCATION = json.load(f)

# 初始化Ray
# address='auto' 會自動連接到 Ray head node
# 如果您在 head node 上運行此腳本，也可以省略 address
# 如果在 worker node 運行並需要連接 head，請使用 ray.init(address=f'ray://{HEAD_NODE_IP}:10001')
try:
    # 嘗試連接已有的 Ray 叢集
    ray.init(address='auto', ignore_reinit_error=True)
    print("Successfully connected to Ray cluster.")
except ConnectionError:
    print(f"Could not connect to existing Ray cluster. Initializing Ray locally.")
    ray.init(ignore_reinit_error=True)


# 訓練函數（CPU）
@ray.remote
def train_model_cpu(num_samples, resource_ip, cpu_cores, num_epochs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        train_dataset = torchvision.datasets.CIFAR10(
            root='/home/ray_cluster/Documents/workspace/tune_population_based', train=True, download=True, transform=transform
        )
    except Exception as e:
        print(f"Error downloading dataset on {resource_ip} (CPU): {e}. Using local root './data'")
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
    )

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)
    device = torch.device("cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    start_time = time.time()
    print(f'Starting CPU task on {resource_ip} (Node: {socket.gethostname()}) with {cpu_cores} cores')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        # 移除 tqdm
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # if (i + 1) % 10 == 0: # 每10個 batch 打印一次日誌
            #     print(f"Node {resource_ip} (CPU) Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    end_time = time.time()
    total_time = end_time - start_time
    throughput = num_samples * num_epochs / total_time
    performance_score = throughput
    print(f'Finished CPU task on {resource_ip}. Time: {total_time:.2f}s, Score: {performance_score:.2f}')

    # resource_ip 為分配的 ip
    return total_time, performance_score, resource_ip, "CPU"

# 訓練函數（GPU）
@ray.remote(num_gpus=1) # 明確指定任務需要1個GPU
def train_model_gpu(num_samples, resource_ip, gpu_count, num_epochs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        train_dataset = torchvision.datasets.CIFAR10(
            root='/home/ray_cluster/Documents/workspace/tune_population_based', train=True, download=True, transform=transform
        )
    except Exception as e:
        print(f"Error downloading dataset on {resource_ip} (GPU): {e}. Using local root './data'")
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
    )

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == 'cpu':
        print(f"Warning: Node {resource_ip} (Actual host: {socket.gethostname()}) was assigned a GPU task but torch.cuda.is_available() is False.")

    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    start_time = time.time()
    print(f'Starting GPU task on {resource_ip} (Node: {socket.gethostname()}) with {gpu_count} GPUs (using device: {device})')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        # 移除 tqdm
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # if (i + 1) % 10 == 0: # 每10個 batch 打印一次日誌
            #     print(f"Node {resource_ip} (GPU) Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    end_time = time.time()
    total_time = end_time - start_time
    throughput = num_samples * num_epochs / total_time
    performance_score = throughput
    print(f'Finished GPU task on {resource_ip}. Time: {total_time:.2f}s, Score: {performance_score:.2f}')

    # resource_ip 為分配的 ip
    return total_time, performance_score, resource_ip, "GPU"

# 設定參數
num_samples = 512  # 每個 batch 的樣本數 (此處命名可能易誤解，但基於原碼保留)
num_epochs = 10  # 訓練10個epoch

# 每個節點分別啟動CPU和GPU的並行訓練任務
tasks = []
# 合併所有節點IP，避免重複
all_node_ips = set(RESOURCE_ALLOCATION["CPU"].keys()) | set(RESOURCE_ALLOCATION["GPU"].keys())

for ip in all_node_ips:
    num_cpus = RESOURCE_ALLOCATION["CPU"].get(ip, 0)
    num_gpus = RESOURCE_ALLOCATION["GPU"].get(ip, 0)

    # --- 關鍵修正：使用節點資源來 "釘選" 任務 ---
    # Ray 節點會自動註冊 "node:IP" 資源 (前提是 Ray 啟動時 IP 正確)
    # 我們可以請求這個資源 (即使是很小量) 來將任務強制分配到該節點
    node_resource_key = f"node:{ip}"

    if num_cpus > 0:
        print(f"Assigning CPU task for node {ip} (requesting {num_cpus} CPUs on {node_resource_key})")
        # 傳入 ip 和 num_cpus
        cpu_task = train_model_cpu.options(
            num_cpus=num_cpus, # 請求 CPU 核心數
            resources={node_resource_key: 0.01} # 請求少量節點資源以"釘選"任務
        ).remote(
            num_samples, ip, num_cpus, num_epochs)
        tasks.append(cpu_task)

    if num_gpus > 0:
        print(f"Assigning GPU task for node {ip} (requesting {num_gpus} GPUs on {node_resource_key})")
        # 傳入 ip 和 num_gpus
        gpu_task = train_model_gpu.options(
            num_gpus=num_gpus, # 請求 GPU 數量
            num_cpus=1, # GPU 任務通常也需要至少 1 個 CPU
            resources={node_resource_key: 0.01} # 請求少量節點資源以"釘選"任務
        ).remote(
            num_samples, ip, num_gpus, num_epochs)
        tasks.append(gpu_task)

print(f"Waiting for {len(tasks)} tasks to complete...")
# 獲取結果
try:
    results = ray.get(tasks)
    print("All tasks completed.")
except Exception as e:
    print(f"Error during ray.get(): {e}")
    print("Some tasks may have failed. Proceeding with completed tasks.")
    # 嘗試獲取已完成的任務結果 (如果需要更精細的錯誤處理)
    # results = []
    # for task in tasks:
    #     try:
    #         results.append(ray.get(task))
    #     except Exception as task_error:
    #         print(f"Task failed: {task_error}")
    results = ray.get(tasks, timeout=1) # 嘗試獲取已完成的


# === 結果處理和儲存 (符合您要求的格式) ===

# 1. 準備新的 JSON 結構
results_json = {
    "CPU": {},
    "GPU": {}
}

# 2. 遍歷 Ray 的結果
for result in results:
    # 確保 result 不是 None (如果任務失敗)
    if result is None:
        continue

    total_time, performance_score, resource_ip, device_type = result

    # 3. 從 RESOURCE_ALLOCATION 獲取核心數
    core_count = 0
    if device_type in RESOURCE_ALLOCATION and resource_ip in RESOURCE_ALLOCATION[device_type]:
        core_count = RESOURCE_ALLOCATION[device_type][resource_ip]

    # 4. 將 performance_score 四捨五入為整數
    int_score = int(round(performance_score))

    # 5. 填充新的 JSON 結構
    if device_type == "CPU":
        results_json["CPU"][resource_ip] = {
            "core": core_count,
            "score": int_score
        }
    elif device_type == "GPU":
        results_json["GPU"][resource_ip] = {
            "core": core_count,
            "score": int_score
        }

# 6. 打印最終結果到控制台
print("\n--- Benchmark Results (New Format) ---")
print(json.dumps(results_json, indent=4))

# 7. 将结果保存到JSON文件
output_file_path = 'score.json'
try:
    # 確保目錄存在
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file_path, 'w') as json_file:
        json.dump(results_json, json_file, indent=4)
    print(f"\nSuccessfully saved results to {output_file_path}")
except Exception as e:
    print(f"\nError saving results to {output_file_path}: {e}")
    print("Saving to local directory as 'HAO_Score_local.json' instead.")
    with open('HAO_Score_local.json', 'w') as json_file:
        json.dump(results_json, json_file, indent=4)

print("Script finished.")
