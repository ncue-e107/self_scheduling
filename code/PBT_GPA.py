import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
from enum import Enum, auto
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from pathlib import Path

import ray
from ray import tune
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from numpy import random
import time
import copy
import os
import heapq
import json
import math
import matplotlib.pyplot as plt
import argparse
import threading

# Trial scheduling chioce
class TrialMode(Enum):
    ERA = "ERA"
    ETA = "ETA"

DATA_PATH = "~/Documents/workspace/tune_population_based/"

DIR_PATH = "~/Documents/workspace/TBS_for_1/"


HEAD_NODE_IP = "PLEASE EXCHANGE YOUR HEAD NODE IP!"     # 頭節點IP
HYPER_NUM = 50                      # 超參數數量
BATCH_SIZE = [32, 64, 128, 256, 512]     # 訓練一個interation的batch size
STOP_ITER = 1000                    # 共訓練幾個iteration
STOP_ACC = 0.8                      # 訓練到準確率停止
INTERVAL_REPORT = 30                # 間隔多久在ternimal中顯示執行過程
INTERVAL_CHECK = 50
STAGE = 100
SLOPE = 0.9
STALENESS = True                  # 是否考慮Staleness

MAX_RETIRE_NODES = 9  # 最大可淘汰節點數量
LATE_STAGE = 0.8  # 進入後期的門檻
RESOURCE_ALLOCATION = {}

with Path("./score.json").open("r") as f:
    RESOURCE_ALLOCATION = json.load(f)
# TEST_SIZE = 25


# 建立data_loader
def get_data_loader(model_type, batch_size = 64, data_dir="~/Documents/workspace/tune_population_based/data"):
    # 強制轉成 Python int，避免 numpy.int64 觸發 DataLoader 檢查錯誤
    batch_size = int(batch_size)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if model_type == "resnet-18":
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root=data_dir, train=True, download=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root=data_dir, train=False, download=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=False,
        )
    elif model_type == "resnet-50":
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(
                root=data_dir, train=True, download=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(
                root=data_dir, train=False, download=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=False
        )
    return train_loader, test_loader
# 模型訓練
def train(model, optimizer, train_loader, device=None):
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    for (inputs, targets) in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        break


def test(model, test_loader, device):
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return correct / total



# 用來顯示當前所有Trail的狀態 (在command中顯示)
@ray.remote(num_cpus = 0.1, resources={"node:"+HEAD_NODE_IP: 0.1})
def Reporter(tuner, max_report_frequency = 5, hyper_num = 1):
    start_time = ray.get(tuner.get_start_time.remote())
    resource = ray.get(tuner.get_resource.remote())
    while True:
        hypers, accuracy, state, perturbs, running_trial_num, package_size = ray.get(tuner.get_for_reporter.remote())
        m, s = divmod(time.time() - start_time, 60)
        h, m = divmod(m, 60)
        if "CPU" in ray.available_resources():
            unused_cpu_num = ray.available_resources()["CPU"]
        else:
            unused_cpu_num = 0
        if "GPU" in ray.available_resources():
            unused_gpu_num = ray.available_resources()["GPU"]
        else:
            unused_gpu_num = 0

        print("== Status ==")
        print(f'Current Time : {time.ctime() } (runnung for {str(int(h)).zfill(2)}:{str(int(m)).zfill(2)}:{str(int(s)).zfill(2)})')
        print(f"Unused Resource : {unused_cpu_num} CPUs and {unused_gpu_num} GPUs")
        print(f"PBT : {perturbs} perturbs")
        print(f'Total hypers : {hyper_num} ( {running_trial_num} is training ), package_size : {package_size}')
        print("+--------------+------------+------------+---------------------+------------+------------+------------+----------+-------+-----------------+")
        print("| Hyper name   |   Status   |  CPU / GPU |                  IP |         lr |   momentum | batch_size |      acc |  iter |   total time (s)|")
        print("+--------------+------------+------------+---------------------+------------+------------+------------+----------+-------+-----------------+")
        for i, (hyper, acc, sta) in enumerate(zip(hypers, accuracy, state)):
            if sta["resource_id"] == -2:
                status = "TERNIMAL"
                cpus_per_trial = 0
                gpus_per_trial = 0
                ip = "node:0.0.0.0"
            elif sta["resource_id"] == -1:
                status = "PENDING"
                cpus_per_trial = 0
                gpus_per_trial = 0
                ip = "node:0.0.0.0"
            else:
                status = "RUNNING"
                cpus_per_trial = resource[sta["resource_id"]]["CPU"]
                gpus_per_trial = resource[sta["resource_id"]]["GPU"]
                ip = resource[sta["resource_id"]]["node"]

            print(f'| hyper_{str(i).zfill(5)}  |   {status:^8}   | {cpus_per_trial:>4.1f} / {gpus_per_trial:<3.1f} | {ip:>19} | {hyper["lr"]:10.6f} | {hyper["momentum"]:10.6f} | {hyper["batch_size"]:>11}| {acc:8.4f} | {sta["iteration"]:>5} | {sta["run_time"]:15.6f} | ')
        print("+--------------+------------+------------+---------------------+------------+------------+------------+----------+-------+-----------------+")
        time.sleep(max_report_frequency)

@ray.remote(num_cpus = 0.2, resources={"node:"+HEAD_NODE_IP: 0.1})
class Tuner(object):
    """
        Tuner : 控制所有Trial進程, 創建與接收Trial的結果。

        Args:
            hyper_num : 共建立多少hyper組合
            batch_siz : 一個iteration訓練的batch大小
            stop_acc : 訓練停止的accuracy條件
            stop_iteration : 訓練停止的iteration條件
            checkpoint_interval : 多少iteration存一個checkpoint
            trials_state : 存每筆hyper訓練使用的資源id，以及訓練花費時間
            resource : 存這樣分配(RESOURCE_ALLOCATION)總共有多少組資源
            avaliable_resource : 存目前可用資源 (正在使用和效能太差就不會在裡面)
            trials_scheduler : 存要訓練的hyper_id (到達終止條件的就不會在裡面)
            running_trial_num : 正在執行訓練的trial數量
            min_run_one_interval_time : 執行一個interval最少需要的時間 (當計算每個資源能力的基礎)

    """
    def __init__(
        self,
        hyper_num = 1,
        model_type = "resnet-18",
        resource_allocation = None,
        trial_allocation = None,
        stop_acc = 1,
        stop_iteration = 0,
        checkpoint_interval = 5,
        hyperparam_mutations = None,
        path = None,
        trialmode = "ERA",
        log_dir = None,  # 新增：模式專屬輸出資料夾
        comm_log_filename=None, # 新增: 通訊時間記錄檔案
        max_retire_nodes=0, # <-- [新增] 接收淘汰數量
    ):
        self.start_time = time.time()
        self.tuner = None
        self.hyper_num = hyper_num
        self.model_type = model_type
        self.stop_acc = stop_acc
        self.stop_iteration = stop_iteration
        self.checkpoint_interval = checkpoint_interval
        self.hyperparam_mutations = hyperparam_mutations
        self.path = path
        self.log_dir = log_dir  # 新增：保存路徑
        self.comm_log_filename = comm_log_filename # 新增: 通訊時間記錄檔案

        self.MAX_RETIRE_NODE_COUNT = max_retire_nodes # <-- [新增] 儲存淘汰數量

        if isinstance(trialmode, TrialMode):
            self.trialmode = trialmode
        else:
            self.trialmode = TrialMode(trialmode.upper())
        self._schedule_fn = self.choice_create_trial(self.trialmode)

        self.trials_scheduler = []
        self.hypers = []
        self.trials_state = []
        self.checkpoints = []
        self.last_checkpoint = [0] * hyper_num
        self.perturbs = 0
        self.trial_acc_list = [0] * hyper_num
        self.resource = []
        self.avaliable_resource = []

        self.running_trial_num = 0
        self.running_resource_num = 0
        self.min_run_one_interval_time = 9999
        self.max_iter = 0
        self.max_acc = -1
        self.last_run_interval = 9999
        self.package_size = 0

        self.start_trial_time = []
        self.resource_run_time = []

        # 新增：紀錄各節點歷史上使用過的 batch size（集合）
        self.node_batch_sizes_history = {}
        self.trial_allocation = trial_allocation
        self.communication_total_cost = 0.0

        # --- [新增] 淘汰節點相關 ---
        self.LATE_STAGE_ITER_THRESHOLD = 0 # 將在 set_placement_group 中設定
        self.weak_nodes_retired_set = set() # 儲存已淘汰的 node_id
        self.weakest_node_ids = set() # 儲存「候選淘汰」的 node_id
        # --- [新增結束] ---

        self.initialize_all_config()
        self.set_placement_group(resource_allocation)

    # --- [新增函式] ---
    def _retire_idle_weak_nodes(self):
        """
        檢查是否進入「後期」，如果是，則從 avaliable_resource 中
        移除所有閒置的「較弱節點」，直到達到 N 的上限。
        """
        # 1. 檢查是否進入「後期」或是否設定了淘汰
        #    (如果 MAX_RETIRE_NODE_COUNT 是 0，或 weakest_node_ids 是空的，就直接返回)
        if self.max_iter < self.LATE_STAGE_ITER_THRESHOLD or not self.weakest_node_ids:
            return

        # 2. 檢查是否已達淘汰上限
        if len(self.weak_nodes_retired_set) >= self.MAX_RETIRE_NODE_COUNT:
            return # 已達上限，不再淘汰

        # 3. 如果進入後期，掃描可用的資源列表
        new_available_resource_list = []
        retired_count_this_round = 0

        for resource_id in self.avaliable_resource:
            # 再次檢查是否已達上限 (可能在迴圈中達到)
            if len(self.weak_nodes_retired_set) >= self.MAX_RETIRE_NODE_COUNT:
                new_available_resource_list.append(resource_id)
                continue # 已達上限，停止檢查，保留剩餘節點

            # 檢查節點是否為「候選弱節點」
            is_candidate = (resource_id in self.weakest_node_ids)

            # 檢查節點是否「已經被淘汰」
            is_already_retired = (resource_id in self.weak_nodes_retired_set)

            if is_candidate and not is_already_retired:
                # 這是候選弱節點，且尚未達淘汰上限 -> 淘汰它
                print(f"--- 進入後期：淘汰閒置的弱節點 {resource_id} (總淘汰: {len(self.weak_nodes_retired_set)+1}/{self.MAX_RETIRE_NODE_COUNT}) ---")
                self.weak_nodes_retired_set.add(resource_id)
                retired_count_this_round += 1
            else:
                # 這不是弱節點，或已達上限，保留它
                new_available_resource_list.append(resource_id)

        # 4. 更新可用的資源列表
        if retired_count_this_round > 0:
            self.avaliable_resource = new_available_resource_list
    # --- [新增結束] ---

    # ==========================================
    # 新增：將 Trial 插入到已排序的 scheduler 中
    # ==========================================
    def insert_trial(self, tid: int) -> None:
        # 注意：原程式碼變數為 trials_state (list of dict) 和 trials_scheduler (list of int)
        new_iteration = self.trials_state[tid]["iteration"]

        left, right = 0, len(self.trials_scheduler)
        while left < right:
            mid = (left + right) // 2
            # 取得中間那個 trial ID 的 iteration 進行比較
            mid_tid = self.trials_scheduler[mid]
            if self.trials_state[mid_tid]["iteration"] <= new_iteration:
                left = mid + 1
            else:
                right = mid
        self.trials_scheduler.insert(left, tid)

    # 新增: create_new_trial 以目前選定的策略排程
    def create_new_trial(self):
        # --- [新增程式碼] ---
        # 每次要分配新任務前，都先檢查並淘汰閒置的弱節點
        self._retire_idle_weak_nodes()
        # --- [新增結束] ---

        # 若沒有可用資源或沒有排程中的 trial，就不動作
        if not self.avaliable_resource or not self.trials_scheduler:
            return
        # 呼叫對應策略（ERA/ETA）
        self._schedule_fn()


    # 初始化每組hyper的值與checkpoint
    def initialize_all_config(self):
        if self.model_type == "resnet-18":
            model = models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, 10)
        elif self.model_type == "resnet-50":
            model = models.resnet50()
            model.fc = nn.Linear(model.fc.in_features, 100)
        optimizer = optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
        )

        bs_list = BATCH_SIZE

        for i in range(self.hyper_num):
            hyper = {
                "lr": random.uniform(0.001, 1),
                "momentum": random.uniform(0.001, 1),
                # 將 numpy.random.choice 的回傳轉成 Python int
                "batch_size": bs_list[i % len(bs_list)],
                "model_type" : self.model_type,
            }
            trial_state = {
                "resource_id" : -1,
                "run_time": 0,
                "iteration" : 0,
            }
            checkpoint = {
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
                "checkpoint_interval" : self.checkpoint_interval,
            }
            self.trials_scheduler.append(i)
            self.hypers.append(hyper)
            self.trials_state.append(trial_state)
            self.checkpoints.append(checkpoint)

    def training_function(config):
        start_time = time.time()
        # 模擬與server溝通的時間
        time.sleep(1)
        end_time = time.time()
        communication_time = end_time - start_time
        tune.report(communication_time=communication_time)

    # 分配每個節點在訓練時能使用的資源數
    def set_placement_group(self, resource_allocation):
        for nodes in ray.nodes():
            if "CPU" in nodes['Resources'] and nodes['NodeManagerAddress'] in resource_allocation["CPU"]:
                if nodes['NodeManagerAddress'] == HEAD_NODE_IP:
                    sub = 1
                else:
                    sub = 0
                sum_cores = nodes['Resources']['CPU']
                # Check if the node exists in the allocation map
                if nodes['NodeManagerAddress'] in resource_allocation["CPU"]:
                    core_alloc = resource_allocation["CPU"][nodes['NodeManagerAddress']].get('core', 1)
                    score_alloc = resource_allocation["CPU"][nodes['NodeManagerAddress']].get('score', 0)
                    while(int(sum_cores / core_alloc)):
                        self.resource.append({
                            "CPU": core_alloc - sub,
                            "GPU": 0,
                            "node":"node:"+nodes['NodeManagerAddress'],
                            "calculate_ability" : score_alloc,
                            "Used_count" : 0.0,
                        })
                        print(self.resource[-1])
                        sub = 0
                        sum_cores -= core_alloc
            if "GPU" in nodes['Resources'] and nodes['NodeManagerAddress'] in resource_allocation["GPU"]:
                sum_gpus = nodes['Resources']['GPU']
                if nodes['NodeManagerAddress'] in resource_allocation["GPU"]:
                    core_alloc = resource_allocation["GPU"][nodes['NodeManagerAddress']].get('core', 1)
                    score_alloc = resource_allocation["GPU"][nodes['NodeManagerAddress']].get('score', 0)
                    while(int(sum_gpus / core_alloc)):
                        self.resource.append({
                            "CPU": 0,
                            "GPU": core_alloc,
                            "node":"node:"+nodes['NodeManagerAddress'],
                            "calculate_ability" : score_alloc,
                            "Used_count" : 0.0,
                        })
                        print(self.resource[-1])
                        sum_gpus -= core_alloc

        for i in range(len(self.resource)):
            self.avaliable_resource.append(i)

        self.start_trial_time = [0] * len(self.resource)
        # 新增: 避免後續使用未初始化 self.resource_run_time
        # 預設給一個較大的值，ETA 印出或試算不會出錯
        self.resource_run_time = [float('inf')] * len(self.resource)

        # --- [修改此處邏輯] ---
        # 1. 定義「後期」的門檻，例如總 iteration 的 80%
        self.LATE_STAGE_ITER_THRESHOLD = self.stop_iteration * LATE_STAGE

        # 2. 建立一個集合 (set) 來追蹤已被淘汰的節點 (已移至 __init__)
        # self.weak_nodes_retired_set = set()

        # 3. 找出 N 個最弱的節點 ID
        self.weakest_node_ids = set() # 儲存「候選淘汰」的節點 ID

        # 檢查 N > 0 且 資源數 > N (如果總節點數 <= N，淘汰就沒意義了)
        if self.MAX_RETIRE_NODE_COUNT > 0 and len(self.resource) > self.MAX_RETIRE_NODE_COUNT:
            # 獲取 (ability, resource_id) 的列表
            # 我們只考慮 ability > 0 的節點
            node_abilities = []
            for i, r in enumerate(self.resource):
                ability = r.get("calculate_ability", 0)
                if ability > 0:
                    node_abilities.append((ability, i)) # 儲存 (分數, 索引)

            # 按照 ability 排序 (由低到高)
            node_abilities.sort(key=lambda x: x[0])

            # 取得 N 個最弱的 resource_id
            num_to_select = min(self.MAX_RETIRE_NODE_COUNT, len(node_abilities))
            weakest_nodes_list = [res_id for ability, res_id in node_abilities[:num_to_select]]
            self.weakest_node_ids = set(weakest_nodes_list)

            print(f"✅ 將在後期淘汰 {len(self.weakest_node_ids)} 個最弱節點 (設定上限: {self.MAX_RETIRE_NODE_COUNT})。")
            print(f"✅ 候選淘汰節點 (Weakest Node IDs): {self.weakest_node_ids}")
        else:
            print(f"ℹ️ MAX_RETIRE_NODE_COUNT 設為 {self.MAX_RETIRE_NODE_COUNT}，不執行節點淘汰。")
        # --- [修改結束] ---

    #################
    # 排程策略選擇🛒 #
    #################

    def choice_create_trial(self, mode):
        if mode == TrialMode.ETA:
            print("⚙️   ETA")
            return self.ETA
        elif mode == TrialMode.ERA:
            print("⚙️   ERA")
            return self.ERA



    #######################
    # 指數式減少策略排程🔢 #
    #######################
    def ERA(self):
        # self.trials_scheduler = sorted(self.trials_scheduler, reverse=False)
        # self.trials_scheduler.sort(key = lambda t: self.trials_state[t]["iteration"])
        remaining_generations = 0
        for trial_state in self.trials_state:
            if trial_state["resource_id"] < 0:
                remaining_generations += math.ceil((self.max_iter - trial_state["iteration"]) / self.checkpoint_interval)
        cc = 1.0

        if not self.trials_scheduler:
            return

        # 遍歷所有可用資源直到耗盡
        while self.avaliable_resource and self.trials_scheduler:
            resource_id = self.avaliable_resource.pop(0)
            n0 = self.resource[resource_id]["calculate_ability"]
            ids, hypers, checkpoints = [], [], []
            n = 0 # Initialize n

            print('-----原本trial數---------')
            print(len(self.trials_scheduler))

            # 指數衰減參數
            slope = SLOPE
            intervals = STAGE

            x = max(0, (self.stop_iteration - self.last_run_interval) // intervals)

            if n0 >= 1:
                n = math.ceil(n0 * (slope ** x))
                n = min(n, len(self.trials_scheduler))
                print('原始T數'); print(n0)
                print('縮短T數'); print(n)

                if n <= 0:
                    self.avaliable_resource.append(resource_id)
                    continue

                for i in range(n):
                    tid = self.trials_scheduler.pop(0)
                    ids.append(tid)
                    if (self.stop_iteration - self.trials_state[tid]["iteration"]) < self.checkpoint_interval:
                        self.checkpoints[tid]["checkpoint_interval"] = self.stop_iteration - self.trials_state[tid]["iteration"]
                    else:
                        self.checkpoints[tid]["checkpoint_interval"] = self.checkpoint_interval
                    hypers.append(self.hypers[tid])
                    checkpoints.append(self.checkpoints[tid])
            else: # n0 is 0 or less
                self.avaliable_resource.append(resource_id)
                continue

            # This block should not be reachable if n is 0
            if n == 0:
                # If for some reason n is 0, put resource back and stop
                self.avaliable_resource.append(resource_id)
                continue

            print('-------分配出去---------')
            print(f"{n}")
            print('-----剩餘trial數---------')
            print(len(self.trials_scheduler))
            r = self.resource_run_time[resource_id]
            if math.isfinite(r) and r > 0:
                print('-----trial時間(估計)---------')
                print(math.ceil(r))
            print('----------------------------')

            # 啟動訓練
            Trial.options(
                num_cpus=self.resource[resource_id]["CPU"],
                num_gpus=self.resource[resource_id]["GPU"],
                resources={self.resource[resource_id]["node"]: 0.1}
            ).remote(self.tuner, n, ids, hypers, checkpoints)

            self.start_trial_time[resource_id] = time.time()
            self.running_trial_num += n
            self.running_resource_num += 1
            self.resource[resource_id]["Used_count"] += cc
            for tid in ids:
                self.trials_state[tid]["resource_id"] = resource_id


    ##########################
    # Trial執行時間策略排程⏲️ #
    ##########################
    def ETA(self):
        # self.trials_scheduler = sorted(self.trials_scheduler, reverse=False)
        # self.trials_scheduler.sort(key = lambda t: self.trials_state[t]["iteration"])

        remaining_generations = 0
        for trial_state in self.trials_state:
            if trial_state["resource_id"] < 0:
                remaining_generations += math.ceil((self.max_iter - trial_state["iteration"]) / self.checkpoint_interval)
        cc = 1.0

        # --- 修正程式碼開始 ---
        # 1. 動態計算基準時間
        # 直接迭代列表 self.resource_run_time
        valid_run_times = [t for t in self.resource_run_time if math.isfinite(t) and t > 0]

        if valid_run_times:
            # 使用當前最慢節點的時間作為基準
            adaptive_baseline_time = max(valid_run_times)
        else:
            # 如果還沒有任何有效的執行時間記錄，使用一個預設值
            adaptive_baseline_time = 150.0
        # --- 修正程式碼結束 ---

        if len(self.trials_scheduler):
            while True:
                if not len(self.avaliable_resource):
                    break

                resource_id = self.avaliable_resource.pop(0)
                n_ability = self.resource[resource_id]["calculate_ability"]
                ids = []
                hypers = []
                checkpoints = []
                n = 0 # Initialize n
                print('-----原本trial數---------')
                print(len(self.trials_scheduler))

                if n_ability >= 1 and self.resource[resource_id]["Used_count"] == 0.0:
                    n = 1
                    for i in range(n):
                        tid = self.trials_scheduler.pop(0)
                        ids.append(tid)
                        if (self.stop_iteration - self.trials_state[tid]["iteration"]) < self.checkpoint_interval:
                            self.checkpoints[tid]["checkpoint_interval"] = self.stop_iteration - self.trials_state[tid]["iteration"]
                        else:
                            self.checkpoints[tid]["checkpoint_interval"] = self.checkpoint_interval
                        hypers.append(self.hypers[tid])
                        checkpoints.append(self.checkpoints[tid])

                elif n_ability >= 1 and self.resource[resource_id]["Used_count"] != 0.0:
                    r = self.resource_run_time[resource_id]
                    if not math.isfinite(r) or r <= 0:
                        n = 1
                    else:
                        # 使用自適應基準時間來取代 150.0
                        n = max(1, math.ceil(adaptive_baseline_time / r))
                    n = min(n, len(self.trials_scheduler))

                    print('秒數/估計單 trial：')
                    print(r)
                    print('縮短T數')
                    print(n)

                    for i in range(n):
                        tid = self.trials_scheduler.pop(0)
                        ids.append(tid)
                        if (self.stop_iteration - self.trials_state[tid]["iteration"]) < self.checkpoint_interval:
                            self.checkpoints[tid]["checkpoint_interval"] = self.stop_iteration - self.trials_state[tid]["iteration"]
                        else:
                            self.checkpoints[tid]["checkpoint_interval"] = self.checkpoint_interval
                        hypers.append(self.hypers[tid])
                        checkpoints.append(self.checkpoints[tid])
                else: # n_ability is 0 or less
                        self.avaliable_resource.append(resource_id)
                        break

                if n == 0:
                    self.avaliable_resource.append(resource_id)
                    break

                print('-------分配出去---------')
                print(f"{n}")
                print('-----剩餘trial數---------')
                print(len(self.trials_scheduler))
                r = self.resource_run_time[resource_id]
                if math.isfinite(r) and r > 0:
                    print('-----單個trial時間(估計)---------')
                    print(math.ceil(r))
                print('----------------------------')

                Trial.options(
                    num_cpus=self.resource[resource_id]["CPU"],
                    num_gpus=self.resource[resource_id]["GPU"],
                    resources={self.resource[resource_id]["node"]: 0.1}
                ).remote(self.tuner, n, ids, hypers, checkpoints)

                self.start_trial_time[resource_id] = time.time()
                self.running_trial_num += n
                self.running_resource_num += 1
                self.resource[resource_id]["Used_count"] += cc
                for i in range(n):
                    self.trials_state[ids[i]]["resource_id"] = resource_id
                break



    # 處理訓練完要結束的trial
    def report_before_trial_end(self, n, ids, accuracys, run_times, checkpoints):
        end_trial_time = time.time()
        total_run_time = 0

        # 暫存 resource_id，因為迴圈中會用到
        # 假設同一次 Trial.remote 的所有 trial 都來自同一個 resource_id
        if not ids: # 如果 ids 是空的，提前返回
            self.create_new_trial()
            return

        resource_id = self.trials_state[ids[0]]["resource_id"]

        for i in range(n):
            self.trial_acc_list[ids[i]] = accuracys[i]

            if checkpoints[i]["checkpoint_interval"] >= self.checkpoint_interval:
                mutation.remote(self.tuner, ids[i], self.hypers, self.trial_acc_list, self.last_checkpoint, self.hyperparam_mutations)

            # resource_id = self.trials_state[ids[i]]["resource_id"] # resource_id 應該是固定的
            self.trials_state[ids[i]]["resource_id"] = -1

            self.trials_state[ids[i]]["run_time"] += run_times[i]
            self.trials_state[ids[i]]["iteration"] += checkpoints[i]["checkpoint_interval"]
            self.checkpoints[ids[i]] = checkpoints[i]

            # 新增：記錄每次實際使用到的節點與對應 batch size
            node = self.resource[resource_id]["node"]
            bs_used = int(self.hypers[ids[i]].get("batch_size", 0))
            self.node_batch_sizes_history.setdefault(node, set()).add(bs_used)

            total_run_time = run_times[i]
            if i == n - 1:
                # 更新該 resource 的「平均單個 trial 時間」估計值，避免 ETA 使用 inf 或 0
                # run_times 是累積時間，因此用 run_times[-1] / n 當作平均每個 trial 的耗時估計
                try:
                    avg_per_trial = max(1e-6, float(run_times[-1]) / max(1, n))
                except Exception:
                    avg_per_trial = float('inf')
                self.resource_run_time[resource_id] = avg_per_trial

                communication_time = end_trial_time - self.start_trial_time[resource_id] - total_run_time
                save_communication_time_to_txt.remote(self.log_dir, self.comm_log_filename, n, self.resource[resource_id],  self.trials_state[ids[i]]["iteration"], communication_time, total_run_time, run_times[i])
            save_acc_to_json.remote(ids[i], accuracys[i], self.trials_state[ids[i]]["iteration"], self.path)

            if self.resource[resource_id]["Used_count"] == 0.5:
                self.min_run_one_interval_time = min(self.min_run_one_interval_time, run_times[i])
                calculate_ability = math.ceil(run_times[i] / self.min_run_one_interval_time)
                for resource in self.resource:
                    if resource["calculate_ability"]:
                        self.resource[resource_id]["calculate_ability"] += int(calculate_ability / resource["calculate_ability"])
                self.resource[resource_id]["calculate_ability"] += calculate_ability
                print(self.resource[resource_id])

            if self.trials_state[ids[i]]["iteration"] > self.max_iter:
                self.max_iter = self.trials_state[ids[i]]["iteration"]
                self.last_run_interval = int((self.stop_iteration - self.max_iter) / self.checkpoint_interval * self.hyper_num)

            if accuracys[i] > self.max_acc:
                self.max_acc = accuracys[i]

            check = 0
            if self.stop_iteration:
                if self.trials_state[ids[i]]["iteration"] < self.stop_iteration:
                    check += 1
                else:
                    check = -9

            if self.stop_acc != 1:
                if accuracys[i] < self.stop_acc:
                    check += 1
                else:
                    check = -9
            if check > 0:
                if STALENESS:
                    self.insert_trial(ids[i])
                else:
                    self.trials_scheduler.append(ids[i])
            elif check < 0:
                self.trials_state[ids[i]]["resource_id"] = -2
            else:
                print("No end condition!!")
                exit(0)

        self.running_trial_num -= n
        self.running_resource_num -= 1

        # --- [修改此處] ---
        # 檢查是否進入「後期」
        is_late_stage = self.max_iter >= self.LATE_STAGE_ITER_THRESHOLD
        # 檢查此節點是否為「候選弱節點」
        is_candidate = (resource_id in self.weakest_node_ids)
        # 檢查是否已達淘汰上限
        under_limit = len(self.weak_nodes_retired_set) < self.MAX_RETIRE_NODE_COUNT

        if is_late_stage and is_candidate and under_limit:
            # 進入後期，是候選弱節點，且尚未達上限 -> 淘汰它
            print(f"--- 進入後期：淘汰剛完成任務的弱節點 {resource_id} (總淘汰: {len(self.weak_nodes_retired_set)+1}/{self.MAX_RETIRE_NODE_COUNT}) ---")
            self.weak_nodes_retired_set.add(resource_id)
            # (重點：不要執行 append)
        else:
            # 非後期，或非候選，或已達上限 -> 正常加回去
            # (我們也檢查它是否"已經"在淘汰名單中，避免意外加回)
            if resource_id not in self.weak_nodes_retired_set:
                self.avaliable_resource.append(resource_id)
        # --- [修改結束] ---

        self.create_new_trial()

    # 新增：取得各節點歷史使用過的 batch size（已排序的列表）
    def get_node_batch_sizes_history(self):
        return {node: sorted(list(s)) for node, s in self.node_batch_sizes_history.items()}


    # 查看是否全部都訓練完
    def is_finish(self):
        if len(self.trials_scheduler) + self.running_trial_num == 0:
            return True
        else:
            return False

    # 設定head的指標
    def set_head(self, tuner):
        self.tuner = tuner

        for _ in range(len(self.resource)):
            self.create_new_trial()

    def set_after_mutation(self, id, chosed_id, new_hyper, last_checkpoint):
        self.last_checkpoint[id] = last_checkpoint

        if new_hyper:
            self.perturbs += 1
            self.hypers[id] = new_hyper
            self.checkpoints[id] = copy.deepcopy(self.checkpoints[chosed_id])

            # if self.trials_state[id]["iteration"] > self.trials_state[chosed_id]["iteration"]:
            #     if self.trials_state[id]["iteration"] == self.stop_iteration:
            #         self.trials_scheduler.append(id)
            #         self.trials_scheduler = sorted(self.trials_scheduler, reverse=False)
            #     self.trials_state[id]["iteration"] = self.trials_state[chosed_id]["iteration"]


    def get_for_reporter(self):
        return self.hypers, self.trial_acc_list, self.trials_state, self.perturbs, self.running_trial_num, self.package_size

    def get_start_time(self):
        return self.start_time

    def get_resource(self):
        return self.resource

    def get_best_accuracy(self):
        max_list = list(map(self.trial_acc_list.index, heapq.nlargest(1, self.trial_acc_list)))
        return max_list[0], self.trial_acc_list[max_list[0]], self.perturbs


# 突變
@ray.remote(num_cpus = 0.1, resources={"node:"+HEAD_NODE_IP: 0.1})
def mutation(tuner, id, hypers, accuracys, last_checkpoint, hyperparam_mutations, resample_posibility = 0.25, quantile_fraction = 0.25):
    lower_quantile, upper_quantile = quantile(accuracys, quantile_fraction)
    if id in upper_quantile:
        last_checkpoint[id] = 1
    else:
        last_checkpoint[id] = 0

    new_hyper = None
    chosed_id = None

    if id in lower_quantile:      # 是否表現很差
        print("--- Exploit ---")
        chosed_id = random.choice(upper_quantile)       # 選出一個優秀的Trial
        print(f"Cloning  hyper_{str(chosed_id).zfill(5)} (score : {accuracys[chosed_id]}) to hyper_{str(id).zfill(5)} (score : {accuracys[id]}) \n")
        if last_checkpoint[chosed_id] == 0:
            print(f"Hyper_{str(chosed_id).zfill(5)} don't have checkpoint, skip exploit for  hyper_{str(id).zfill(5)}!!")
        else:
            new_hyper = explore(id, hypers[chosed_id],  hyperparam_mutations, resample_posibility)      # 突變

    tuner.set_after_mutation.remote(id, chosed_id, new_hyper, last_checkpoint[id])

# 找出標線優秀跟差的
def quantile(accuracys, quantile_fraction):
    trials = []
    for id, acc in enumerate(accuracys):
        if acc != 0:
            trials.append(id)

    if len(trials) <= 1:
        return [], []

    trials.sort(key=lambda t: accuracys[t])

    # 計算num_trials_in_quantile
    num_trials_in_quantile = int(math.ceil(len(trials) * quantile_fraction))
    if num_trials_in_quantile > len(trials) / 2:
        num_trials_in_quantile = int(math.floor(len(trials) / 2))

    return (trials[:num_trials_in_quantile], trials[-num_trials_in_quantile:])

# 探索新的hyper
def explore(id, hyper, hyperparam_mutations, resample_posibility):
    new_hyper = hyper
    print(f"--- Explore the hyperparameters on  hyper_{str(id).zfill(5)} ---")
    for key, distribution in hyperparam_mutations.items():
        print(f'{key} : {hyper[key]} --- ', end="")
        if isinstance(distribution, list):
            if random.random() < resample_posibility or hyper[key] not in distribution:
                val = random.choice(distribution)
                # 若是 batch_size，確保轉成 Python int
                new_hyper[key] = int(val) if key == "batch_size" else val
                print(f'(resample)   --> {new_hyper[key]}')
            else:
                shift = random.choice([-1, 1])
                old_idx = distribution.index(int(hyper[key]) if key == "batch_size" else hyper[key])
                new_idx = min(max(old_idx + shift, 0), len(distribution) - 1)
                val = distribution[new_idx]
                new_hyper[key] = int(val) if key == "batch_size" else val
                print(f"(shift {'left' if shift == -1 else 'right'}) --> {new_hyper[key]}")
        elif isinstance(distribution, tuple):
            if random.random() < resample_posibility:
                new_hyper[key] = random.uniform(distribution[0], distribution[1])
                print(f'(resample)   --> {new_hyper[key]}')
            else:
                mul = random.choice([0.8, 1.2])
                new_hyper[key] = hyper[key] * mul
                print(f'(* {mul})   --> {new_hyper[key]}')
    print()
    bs_list = BATCH_SIZE
    new_hyper["batch_size"] = bs_list[id % len(bs_list)]
    return new_hyper

@ray.remote(num_cpus = 0.1, resources={"node:"+HEAD_NODE_IP: 0.1})
def save_acc_to_json(id, acc, iter, path):
    jsonFile = open(path+'/'+str(id)+'-accuracy.json','a')
    data={
        "iteration":iter,
        "accuracy":acc,
    }
    w = json.dumps(data)    # 產生要寫入的資料
    jsonFile.write(w)       # 寫入資料
    jsonFile.write('\n')    # 寫入資料
    jsonFile.close()

@ray.remote(num_cpus=0.1, resources={"node:" + HEAD_NODE_IP: 0.1})
def save_communication_time_to_txt(log_dir, filename, trial_num, resources, iter, communication_time, total_run_time, run_times):
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, filename)
    with open(out_path, 'a') as communication_file:
        communication_file.write(f"resources : {resources}\n")
        communication_file.write(f"iter: {iter}\n")
        communication_file.write(f"num: {trial_num}\n")
        communication_file.write(f"trial_sec: {run_times}\n")
        communication_file.write(f"communication_time: {communication_time:.2f}\n")
        communication_file.write(f"total_run_time: {total_run_time:.2f}\n")
        communication_file.write(f"-----------------------------------------------\n")

# 會被分配一個hyper，設計訓練與data傳接
@ray.remote
def Trial(tuner, n, ids, hypers, checkpoints):
    # Check for empty hypers to prevent crash
    if not hypers:
        return

    start_time = time.time()
    accs = []
    run_times = []

    model_type = hypers[0].get("model_type", "resnet-18")

    if model_type == "resnet-18":
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif model_type == "resnet-50":
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, 100)

    # 依 batch_size 快取不同的 DataLoader
    loaders_cache = {}  # key: (model_type, batch_size) -> (train_loader, test_loader)

    for i in range(n):
        # 強制轉 Python int（避免 numpy.int64）
        bs = int(hypers[i].get("batch_size", 512))
        key = (model_type, bs)
        if key not in loaders_cache:
            loaders_cache[key] = get_data_loader(model_type, bs)
        train_loader, test_loader = loaders_cache[key]

        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = model.to(device)
            # Make sure checkpoint tensors are on the correct device
            for k, v in checkpoints[i]["model_state_dict"].items():
                checkpoints[i]["model_state_dict"][k] = v.to(device)
            for state in checkpoints[i]["optimizer_state_dict"]["state"].values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        else:
            device = torch.device("cpu")

        model.load_state_dict(checkpoints[i]["model_state_dict"])
        optimizer = optim.SGD(model.parameters(), lr=hypers[i].get("lr", 0.01), momentum=hypers[i].get("momentum", 0.9))
        optimizer.load_state_dict(checkpoints[i]["optimizer_state_dict"])

        for param_group in optimizer.param_groups:
            if "lr" in hypers[i]:
                param_group["lr"] = hypers[i]["lr"]
            if "momentum" in hypers[i]:
                param_group["momentum"] = hypers[i]["momentum"]

        for _ in range(checkpoints[i]["checkpoint_interval"]):
            train(model, optimizer, train_loader, device)

        accs.append(test(model, test_loader, device))
        run_times.append(time.time() - start_time)

        # Move state dicts back to CPU before sending them back to the Tuner
        checkpoints[i]["model_state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}

        cpu_opt_state = copy.deepcopy(optimizer.state_dict())
        for state in cpu_opt_state['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        checkpoints[i]["optimizer_state_dict"] = cpu_opt_state


    tuner.report_before_trial_end.remote(n, ids, accs, run_times, checkpoints)

def allocate_trials_by_score(score_json_data, hyper_num):
    # 1. 蒐集所有節點分數（<=0 強制設為1），使用唯一的鍵來區分不同資源組
    score_dict = {}
    for node_type in ['CPU', 'GPU']:
        if node_type in score_json_data:
            for ip, info in score_json_data[node_type].items():
                # 使用 IP 和資源類型作為唯一的鍵
                key = f"{ip}_{node_type}"
                raw_score = info['score']
                score_dict[key] = max(1, raw_score)  # <=0 強制設為1

    # 如果沒有任何節點，直接返回
    if not score_dict:
        return {}

    # 2. 確保每個節點至少分配一個 trial
    allocation = {key: 1 for key in score_dict}
    num_nodes = len(score_dict)

    # 如果 trials 總數小於節點數，則部分節點可能無法分配到
    if hyper_num < num_nodes:
        # 分數高的優先分配
        sorted_keys = sorted(score_dict.keys(), key=lambda k: score_dict[k], reverse=True)
        allocation = {key: 0 for key in score_dict}
        for i in range(hyper_num):
            allocation[sorted_keys[i]] = 1
        return allocation

    remaining_trials = hyper_num - num_nodes

    # 3. 依分數比例分配剩餘的 trial
    # 只計算有資格獲得額外 trial 的節點的總分
    total_score_for_remaining = sum(score_dict.values())

    if total_score_for_remaining > 0:
        for key, score in score_dict.items():
            extra_trials = int(round(score / total_score_for_remaining * remaining_trials))
            allocation[key] += extra_trials

    # 4. 修正 allocation，確保總數等於 hyper_num
    allocated = sum(allocation.values())
    if allocated < hyper_num:
        # trial不夠，依序分給分數最高的
        sorted_keys = sorted(score_dict.keys(), key=lambda k: score_dict[k], reverse=True)
        remain = hyper_num - allocated
        idx = 0
        while remain > 0:
            allocation[sorted_keys[idx % len(sorted_keys)]] += 1
            remain -= 1
            idx += 1
    elif allocated > hyper_num:
        # trial太多，從分數最低的砍掉（但不能砍到1以下）
        sorted_keys = sorted(score_dict.keys(), key=lambda k: score_dict[k])
        remain = allocated - hyper_num
        idx = 0
        while remain > 0:
            key = sorted_keys[idx % len(sorted_keys)]
            if allocation[key] > 1:
                allocation[key] -= 1
                remain -= 1
            idx += 1

    return allocation

if __name__=='__main__':
    parser = argparse.ArgumentParser(description = "Choose trial scheduling mode")
    parser.add_argument("--exp_times", type=int, default=1)
    parser.add_argument("--mode", type=str, default="ERA")
    args = parser.parse_args()
    if args.mode.upper() not in ["ETA", "ERA"]:
        raise KeyError(f"{args.mode} is not exists.")

    data_path = DATA_PATH
    dir_path= DIR_PATH

    # 新增：依模式建立獨立的輸出資料夾
    MODE_NAME = args.mode.upper()
    LOG_DIR = os.path.join(dir_path, 'log_' + MODE_NAME)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 為這次完整的腳本執行生成一個唯一的通訊日誌文件名
    current_time_str = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(LOG_DIR, 'communication_time'), exist_ok=True)
    comm_log_filename = f"communication_time/communication_results_{current_time_str}.txt"

    score_json_file = open(os.path.join(dir_path, 'score.json'),'r')
    score_json_data_original = json.load(score_json_file)
    score_json_file.close()

    trial_allocation = allocate_trials_by_score(score_json_data_original, HYPER_NUM)
    print("[Trial Allocation by Score]", trial_allocation)

    new_json = {"CPU": {}, "GPU": {}}
    for key, num_trials in trial_allocation.items():
        ip, node_type = key.split('_')
        # 確保 score_json_data_original 中有對應的 core 資訊
        if ip in score_json_data_original.get(node_type, {}):
            if ip not in new_json[node_type]:
                new_json[node_type][ip] = {"core": score_json_data_original[node_type][ip]["core"], "score": 0}
            new_json[node_type][ip]["score"] += num_trials

    # 輸出到檔案
    score_json_file_path = os.path.join(dir_path, 'temp_score.json')
    with open(score_json_file_path, 'w') as f:
        json.dump(new_json, f, indent=4)
    print(f"Trial allocation saved to {score_json_file_path}")

    # Tuner 將使用這個新的分配檔
    score_json_data = new_json

    runtime_env = {
        'working_dir': data_path,
        'excludes': ["data/", "my_model/", "ray_results/", "pytorch-cifar/"],
    }

    # 改寫：Running_Results.txt 放到 LOG_DIR 下面
    with open(os.path.join(LOG_DIR, "Running_Results.txt"), "a+") as out_result:
        out_result.write("+---------------+---------------+\n")
        out_result.write(f'{time.ctime()}  <<Our Results - {__file__}>> \n')
        out_result.write(f"Hyper_num = {HYPER_NUM} \n")
        out_result.write(f"Stop iteration = {STOP_ITER} \n")
        out_result.write(f"Stop accuracy = {STOP_ACC} \n")
        out_result.write(f"Checkpoint interval = {INTERVAL_CHECK} \n")
        out_result.write(f"Batch size = {BATCH_SIZE} \n")

        # --- [新增] 寫入淘汰節點設定 ---
        out_result.write(f"Max Retire Nodes = {MAX_RETIRE_NODES} \n")
        # --- [新增結束] ---

        out_result.write(f"Resource allocation: {RESOURCE_ALLOCATION} \n")

    model_types = ["resnet-18"]

    for model in model_types:
        with open(os.path.join(LOG_DIR, "Running_Results.txt"), "a+") as out_result:
            out_result.write(f"model_type: {model} \n")

        avg_run_time = 0
        avg_accuracy = 0

        for i in range(args.exp_times):
            if ray.is_initialized():
                ray.shutdown()
            ray.init(address="ray://"+HEAD_NODE_IP+":10001", runtime_env=runtime_env)
            print(ray.available_resources())

            tt = time.ctime()
            tt_tmp = tt.split()
            json_path = os.path.join(LOG_DIR, "results", f"{tt_tmp[-1]}-{tt_tmp[-4]}-{tt_tmp[-3]}-{tt_tmp[-2]}_run{i+1}/")
            os.makedirs(json_path, exist_ok=True)
            print(f'{json_path = }')

            # 建立Tuner，並傳入 log_dir 和共享的 comm_log_filename
            tuner_head = Tuner.remote(
                hyper_num = HYPER_NUM,
                model_type = model,
                resource_allocation = score_json_data,      # 必須是原始格式，包含 "CPU"、"GPU"
                stop_acc = STOP_ACC,
                stop_iteration = STOP_ITER,
                checkpoint_interval = INTERVAL_CHECK,
                path = json_path,
                hyperparam_mutations = {
                    "lr": (0.0001, 1),
                    "momentum": (0.0001, 1),
                    "batch_size": (BATCH_SIZE)
                },
                trialmode = MODE_NAME,
                log_dir = LOG_DIR,
                comm_log_filename = comm_log_filename, # 傳入本次執行共享的唯一檔案名
                max_retire_nodes = MAX_RETIRE_NODES, # <-- [新增] 傳入參數
            )

            tuner_head.set_head.remote(tuner_head)

            Reporter.remote(
                tuner_head,
                max_report_frequency = INTERVAL_REPORT,
                hyper_num = HYPER_NUM,
            )

            while(not ray.get(tuner_head.is_finish.remote())):
                time.sleep(1)

            max_acc_index, max_acc, perturbs = ray.get(tuner_head.get_best_accuracy.remote())
            start_time = ray.get(tuner_head.get_start_time.remote())
            avg_run_time += (time.time() - start_time)
            avg_accuracy += max_acc
            resource = ray.get(tuner_head.get_resource.remote())

            # 取得各節點歷史使用的 batch size 並寫入 Running_Results.txt
            node_bs = ray.get(tuner_head.get_node_batch_sizes_history.remote())
            with open(os.path.join(LOG_DIR, "Running_Results.txt"), "a+") as out_result:
                out_result.write(f"\n--- Results for experiment run {i+1} ---\n")
                out_result.write("Batch size per node (historical usage):\n")
                for node, sizes in sorted(node_bs.items()):
                    out_result.write(f"  {node}: {sizes}\n")
                out_result.write(f"Resource results: {resource} \n")
            ray.shutdown()
            time.sleep(10)

        with open(os.path.join(LOG_DIR, "Running_Results.txt"), "a+") as out_result:
            out_result.write(f"\n--- Final Average Results ---\n")
            out_result.write(f"Avg_total_runtime : {avg_run_time/args.exp_times} \n")
            out_result.write(f"Avg_accuracy : {avg_accuracy/args.exp_times} \n\n")
