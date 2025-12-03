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

DIR_PATH = "~/Documents/workspace/TBS/"


HEAD_NODE_IP = "PLEASE EXCHANGE YOUR HEAD NODE IP!"      # 頭節點IP
HYPER_NUM = 50                      # 超參數數量
BATCH_SIZE = [32, 64, 128, 256, 512]     # 訓練一個interation的batch size
STOP_ITER = 1000                    # 共訓練幾個iteration
STOP_ACC = 0.8                      # 訓練到準確率停止
INTERVAL_REPORT = 30               # 間隔多久在ternimal中顯示執行過程
INTERVAL_CHECK = 50
STAGE = 100
SLOPE = 0.9
STALENESS = True                  # 是否考慮Staleness
RESOURCE_ALLOCATION = {}

# Use a try-except block to handle potential FileNotFoundError
try:
    with Path("./score.json").open("r") as f:
        RESOURCE_ALLOCATION = json.load(f)
except FileNotFoundError:
    print("Warning: score.json not found. Using empty RESOURCE_ALLOCATION.")
    RESOURCE_ALLOCATION = {}
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
        try:
            hypers, accuracy, state, perturbs, running_trial_num, package_size = ray.get(tuner.get_for_reporter.remote())
            m, s = divmod(time.time() - start_time, 60)
            h, m = divmod(m, 60)

            available_resources = ray.available_resources()
            unused_cpu_num = available_resources.get("CPU", 0)
            unused_gpu_num = available_resources.get("GPU", 0)

            print("== Status ==")
            print(f'Current Time : {time.ctime() } (runnung for {str(int(h)).zfill(2)}:{str(int(m)).zfill(2)}:{str(int(s)).zfill(2)})')
            print(f"Unused Resource : {unused_cpu_num} CPUs and {unused_gpu_num} GPUs")
            print(f"PBT : {perturbs} perturbs")
            print(f'Total hypers : {hyper_num} ( {running_trial_num} is training ), package_size : {package_size}')
            print("+--------------+------------+------------+---------------------+------------+------------+------------+----------+-------+-----------------+")
            print("| Hyper name   |    Status    |  CPU / GPU |                     IP |           lr |   momentum | batch_size |      acc |  iter |    total time (s)|")
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
                    # Check if resource_id is valid before accessing
                    if 0 <= sta["resource_id"] < len(resource):
                        cpus_per_trial = resource[sta["resource_id"]]["CPU"]
                        gpus_per_trial = resource[sta["resource_id"]]["GPU"]
                        ip = resource[sta["resource_id"]]["node"]
                    else:
                        cpus_per_trial = 'N/A'
                        gpus_per_trial = 'N/A'
                        ip = 'Unknown'


                print(f'| hyper_{str(i).zfill(5)}  |  {status:^8}  | {str(cpus_per_trial):>4} / {str(gpus_per_trial):<3} | {ip:>19} | {hyper["lr"]:10.6f} | {hyper["momentum"]:10.6f} | {hyper["batch_size"]:>11}| {acc:8.4f} | {sta["iteration"]:>5} | {sta["run_time"]:15.6f} | ')
            print("+--------------+------------+------------+---------------------+------------+------------+------------+----------+-------+-----------------+")
            time.sleep(max_report_frequency)
        except ray.exceptions.RayActorError as e:
            print(f"Reporter: Tuner actor seems to have exited. Shutting down. Error: {e}")
            break
        except Exception as e:
            print(f"An error occurred in Reporter: {e}")
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
        self.initialize_all_config()
        self.set_placement_group(resource_allocation)

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
        if not resource_allocation:
            print("Warning: resource_allocation is empty or None.")
            return

        for nodes in ray.nodes():
            node_ip = nodes['NodeManagerAddress']
            # Handle CPU resources
            if "CPU" in nodes['Resources'] and "CPU" in resource_allocation and node_ip in resource_allocation["CPU"]:
                sub = 1 if node_ip == HEAD_NODE_IP else 0
                sum_cores = nodes['Resources']['CPU']
                core_alloc = resource_allocation["CPU"][node_ip].get('core', 1)
                score_alloc = resource_allocation["CPU"][node_ip].get('score', 0)

                # Only allocate if score is positive
                if score_alloc > 0:
                    while int(sum_cores / core_alloc) > 0:
                        self.resource.append({
                            "CPU": core_alloc - sub,
                            "GPU": 0,
                            "node": "node:" + node_ip,
                            "calculate_ability": score_alloc,
                            "Used_count": 0.0,
                        })
                        print(self.resource[-1])
                        sub = 0
                        sum_cores -= core_alloc

            # Handle GPU resources
            if "GPU" in nodes['Resources'] and "GPU" in resource_allocation and node_ip in resource_allocation["GPU"]:
                sum_gpus = nodes['Resources']['GPU']
                core_alloc = resource_allocation["GPU"][node_ip].get('core', 1)
                score_alloc = resource_allocation["GPU"][node_ip].get('score', 0)

                # Only allocate if score is positive
                if score_alloc > 0:
                    while int(sum_gpus / core_alloc) > 0:
                        self.resource.append({
                            "CPU": 0,
                            "GPU": core_alloc,
                            "node": "node:" + node_ip,
                            "calculate_ability": score_alloc,
                            "Used_count": 0.0,
                        })
                        print(self.resource[-1])
                        sum_gpus -= core_alloc

        for i in range(len(self.resource)):
            self.avaliable_resource.append(i)

        self.start_trial_time = [0] * len(self.resource)
        # 新增: 避免後續使用未初始化 self.resource_run_time
        # 預設給一個較大的值，ETA 印出或試算不會出錯
        self.resource_run_time = [float('inf')] * len(self.resource)

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
        # if SORT_OR_NOT:
        #     self.trials_scheduler = sorted(self.trials_scheduler, reverse=False)
        #     self.trials_scheduler.sort(key=lambda t: self.trials_state[t]["iteration"])

        # Loop while there are available resources AND trials to be scheduled
        while self.avaliable_resource and self.trials_scheduler:
            resource_id = self.avaliable_resource.pop(0)
            n0 = self.resource[resource_id]["calculate_ability"]

            print('-----原本trial數---------')
            print(len(self.trials_scheduler))

            # If the resource has no calculation ability, put it back and try another one
            if n0 < 1:
                self.avaliable_resource.append(resource_id)
                continue

            # Exponential decay parameter
            slope = SLOPE
            intervals = STAGE
            x = max(0, (self.stop_iteration - self.last_run_interval) // intervals)

            n = math.ceil(n0 * (slope ** x))
            n = min(n, len(self.trials_scheduler))

            print('原始T數'); print(n0)
            print('縮短T數'); print(n)

            # If after calculation, n is 0, it means no trial can be launched with this resource now.
            # Put the resource back and stop trying for this scheduling round.
            if n <= 0:
                self.avaliable_resource.append(resource_id)
                # Since we popped a resource and now we are putting it back,
                # it's better to break the loop to avoid potential infinite loops
                # if all available resources result in n=0. The scheduler will be called again.
                break

            ids, hypers, checkpoints = [], [], []
            for _ in range(n):
                tid = self.trials_scheduler.pop(0)
                ids.append(tid)
                if (self.stop_iteration - self.trials_state[tid]["iteration"]) < self.checkpoint_interval:
                    self.checkpoints[tid]["checkpoint_interval"] = self.stop_iteration - self.trials_state[tid]["iteration"]
                else:
                    self.checkpoints[tid]["checkpoint_interval"] = self.checkpoint_interval
                hypers.append(self.hypers[tid])
                checkpoints.append(self.checkpoints[tid])

            print('-------分配出去---------')
            print(f"{n}")
            print('-----剩餘trial數---------')
            print(len(self.trials_scheduler))
            r = self.resource_run_time[resource_id]
            if math.isfinite(r) and r > 0:
                print('-----trial時間(估計)---------')
                print(math.ceil(r))
            print('----------------------------')

            # Launch the trial
            Trial.options(
                num_cpus=self.resource[resource_id]["CPU"],
                num_gpus=self.resource[resource_id]["GPU"],
                resources={self.resource[resource_id]["node"]: 0.1}
            ).remote(self.tuner, n, ids, hypers, checkpoints)

            self.start_trial_time[resource_id] = time.time()
            self.running_trial_num += n
            self.running_resource_num += 1
            self.resource[resource_id]["Used_count"] += 1.0
            for tid in ids:
                self.trials_state[tid]["resource_id"] = resource_id




    ##########################
    # Trial執行時間策略排程⏲️ #
    ##########################
    def ETA(self):
        # if SORT_OR_NOT:
        #     self.trials_scheduler = sorted(self.trials_scheduler, reverse=False)
        #     self.trials_scheduler.sort(key = lambda t: self.trials_state[t]["iteration"])

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
    # [MODIFIED] Added comm_start_time argument
    def report_before_trial_end(self, n, ids, accuracys, run_times, checkpoints, comm_start_time):
        # [NEW] Calculate communication time based on Script B's logic
        end_trial_time = time.time()
        communication_time = end_trial_time - comm_start_time

        total_run_time = 0
        resource_id = -1 # Should be same for all trials in this batch

        for i in range(n):
            self.trial_acc_list[ids[i]] = accuracys[i]

            if checkpoints[i]["checkpoint_interval"] >= self.checkpoint_interval:
                mutation.remote(self.tuner, ids[i], self.hypers, self.trial_acc_list, self.last_checkpoint, self.hyperparam_mutations)

            resource_id = self.trials_state[ids[i]]["resource_id"]
            self.trials_state[ids[i]]["resource_id"] = -1

            self.trials_state[ids[i]]["run_time"] += run_times[i]
            self.trials_state[ids[i]]["iteration"] += checkpoints[i]["checkpoint_interval"]
            self.checkpoints[ids[i]] = checkpoints[i]

            # 新增：記錄每次實際使用到的節點與對應 batch size
            if 0 <= resource_id < len(self.resource):
                node = self.resource[resource_id]["node"]
                bs_used = int(self.hypers[ids[i]].get("batch_size", 0))
                self.node_batch_sizes_history.setdefault(node, set()).add(bs_used)

            total_run_time = run_times[i] # This is cumulative time, so the last one is total for this trial

            save_acc_to_json.remote(ids[i], accuracys[i], self.trials_state[ids[i]]["iteration"], self.path)

            if 0 <= resource_id < len(self.resource) and self.resource[resource_id]["Used_count"] == 0.5: # This seems to be a specific logic, keeping as is
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

        # This block should be outside the loop and executed once per report.
        if resource_id != -1 and 0 <= resource_id < len(self.resource):
            # Update the resource's average single trial time estimate
            try:
                # Use the total runtime for the batch divided by the number of trials
                avg_per_trial = max(1e-6, float(run_times[-1]) / max(1, n))
            except Exception:
                avg_per_trial = float('inf')
            self.resource_run_time[resource_id] = avg_per_trial

            # [MODIFIED] Use the new 'communication_time' calculated at the top.
            # The old calculation:
            # communication_time = end_trial_time - self.start_trial_time[resource_id] - total_run_time
            save_communication_time_to_txt.remote(
                self.log_dir,
                self.comm_log_filename,
                n,
                self.resource[resource_id],
                self.trials_state[ids[-1]]["iteration"],
                communication_time, # [MODIFIED] Using the new value
                total_run_time,
                run_times[-1]
            )

            self.running_trial_num -= n
            self.running_resource_num -= 1
            self.avaliable_resource.append(resource_id)

        # After processing the results and freeing up a resource, try to schedule new trials.
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
        if not self.trial_acc_list:
             return -1, 0, self.perturbs
        max_acc = max(self.trial_acc_list)
        max_index = self.trial_acc_list.index(max_acc)
        return max_index, max_acc, self.perturbs


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

    if id in lower_quantile and upper_quantile:      # 是否表現很差且有表現好的可以抄
        print("--- Exploit ---")
        chosed_id = random.choice(upper_quantile)       # 選出一個優秀的Trial
        print(f"Cloning  hyper_{str(chosed_id).zfill(5)} (score : {accuracys[chosed_id]}) to hyper_{str(id).zfill(5)} (score : {accuracys[id]}) \n")
        if last_checkpoint[chosed_id] == 0:
            print(f"Hyper_{str(chosed_id).zfill(5)} don't have checkpoint, skip exploit for  hyper_{str(id).zfill(5)}!!")
        else:
            new_hyper = explore(id, hypers[chosed_id],  hyperparam_mutations, resample_posibility)       # 突變

    tuner.set_after_mutation.remote(id, chosed_id, new_hyper, last_checkpoint[id])

# 找出標線優秀跟差的
def quantile(accuracys, quantile_fraction):
    trials_with_acc = [id for id, acc in enumerate(accuracys) if acc > 0]

    if len(trials_with_acc) <= 1:
        return [], []

    trials_with_acc.sort(key=lambda t: accuracys[t])

    num_trials_in_quantile = int(math.ceil(len(trials_with_acc) * quantile_fraction))
    if num_trials_in_quantile > len(trials_with_acc) / 2:
        num_trials_in_quantile = int(math.floor(len(trials_with_acc) / 2))

    if num_trials_in_quantile == 0 and len(trials_with_acc) > 1: # Ensure at least one in each quantile if possible
        num_trials_in_quantile = 1

    return (trials_with_acc[:num_trials_in_quantile], trials_with_acc[-num_trials_in_quantile:])

# 探索新的hyper
def explore(id, hyper, hyperparam_mutations, resample_posibility):
    new_hyper = copy.deepcopy(hyper)
    print(f"--- Explore the hyperparameters on  hyper_{str(id).zfill(5)} ---")
    for key, distribution in hyperparam_mutations.items():
        print(f'{key} : {new_hyper[key]} --- ', end="")
        if isinstance(distribution, list) or isinstance(distribution, tuple) and not isinstance(distribution[0], (int, float)):
                # Categorical parameter
            if random.random() < resample_posibility or new_hyper[key] not in distribution:
                val = random.choice(distribution)
                new_hyper[key] = int(val) if key == "batch_size" else val
                print(f'(resample)  --> {new_hyper[key]}')
            else:
                current_val_int = int(new_hyper[key]) if key == "batch_size" else new_hyper[key]
                # Ensure distribution contains integers if comparing with int
                distribution_int = [int(v) for v in distribution]
                if current_val_int in distribution_int:
                    old_idx = distribution_int.index(current_val_int)
                    shift = random.choice([-1, 1])
                    new_idx = min(max(old_idx + shift, 0), len(distribution) - 1)
                    val = distribution[new_idx]
                    new_hyper[key] = int(val) if key == "batch_size" else val
                    print(f"(shift {'left' if shift == -1 else 'right'}) --> {new_hyper[key]}")
                else:
                    # Fallback to resampling if current value not in list
                    val = random.choice(distribution)
                    new_hyper[key] = int(val) if key == "batch_size" else val
                    print(f'(resample, fallback)  --> {new_hyper[key]}')

        elif isinstance(distribution, tuple): # Continuous parameter
            if random.random() < resample_posibility:
                new_hyper[key] = random.uniform(distribution[0], distribution[1])
                print(f'(resample)  --> {new_hyper[key]}')
            else:
                mul = random.choice([0.8, 1.2])
                new_val = new_hyper[key] * mul
                # Clamp the new value to be within the defined bounds
                new_hyper[key] = max(distribution[0], min(new_val, distribution[1]))
                print(f'(* {mul})  --> {new_hyper[key]}')
    print()
    bs_list = BATCH_SIZE
    new_hyper["batch_size"] = bs_list[id % len(bs_list)]
    return new_hyper

@ray.remote(num_cpus = 0.1, resources={"node:"+HEAD_NODE_IP: 0.1})
def save_acc_to_json(id, acc, iter, path):
    try:
        with open(os.path.join(path, f'{id}-accuracy.json'),'a') as jsonFile:
            data={"iteration":iter, "accuracy":acc}
            w = json.dumps(data)
            jsonFile.write(w + '\n')
    except Exception as e:
        print(f"Could not write to accuracy file for trial {id}: {e}")

@ray.remote(num_cpus=0.1, resources={"node:" + HEAD_NODE_IP: 0.1})
def save_communication_time_to_txt(log_dir, filename, trial_num, resources, iter, communication_time, total_run_time, run_times):
    try:
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
    except Exception as e:
        print(f"Could not write to communication time file: {e}")

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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Ensure checkpoint is loaded to the correct device
        model_state_dict = {k: v.to(device) for k, v in checkpoints[i]["model_state_dict"].items()}
        model.load_state_dict(model_state_dict)

        optimizer = optim.SGD(model.parameters(), lr=hypers[i].get("lr", 0.01), momentum=hypers[i].get("momentum", 0.9))

        # Move optimizer state to the correct device
        opt_state_dict = checkpoints[i]["optimizer_state_dict"]
        for state in opt_state_dict['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        optimizer.load_state_dict(opt_state_dict)

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

    # [MODIFIED] Record start time for communication
    comm_start_time = time.time()

    # [MODIFIED] Pass comm_start_time to the tuner
    tuner.report_before_trial_end.remote(n, ids, accs, run_times, checkpoints, comm_start_time)

def allocate_trials_by_score(score_json_data, hyper_num):
    """
    Allocates trials to nodes based on their score.
    Nodes with a score of 0 or less are not allocated any trials.
    """
    # 1. Collect all nodes from the original data to handle the fallback case
    all_nodes = {}
    for node_type in ['CPU', 'GPU']:
        if node_type in score_json_data:
            for ip, info in score_json_data[node_type].items():
                key = f"{ip}_{node_type}"
                all_nodes[key] = info.get('score', 0)

    # 2. Filter for nodes with a score > 0
    eligible_nodes = {key: score for key, score in all_nodes.items() if score > 0}

    # If no nodes have a positive score, distribute evenly among all available nodes as a fallback.
    if not eligible_nodes:
        print("Warning: No nodes with a positive score found. Distributing trials evenly among all nodes.")
        target_nodes = all_nodes if all_nodes else {}
        if not target_nodes:
            print("Error: No nodes found to allocate trials to.")
            return {}

        allocation = {}
        num_nodes = len(target_nodes)
        base_alloc = hyper_num // num_nodes
        remainder = hyper_num % num_nodes

        for i, key in enumerate(sorted(target_nodes.keys())):
            allocation[key] = base_alloc + (1 if i < remainder else 0)
        return allocation

    # 3. Allocate trials proportionally to eligible nodes
    total_score = sum(eligible_nodes.values())
    allocation = {key: 0 for key in eligible_nodes}

    for key, score in eligible_nodes.items():
        allocation[key] = int(score / total_score * hyper_num)

    # 4. Correct the allocation to ensure the total number of trials is exactly hyper_num
    allocated_sum = sum(allocation.values())
    remainder = hyper_num - allocated_sum

    if remainder > 0:
        # If trials are missing, add them one by one to the highest-scored nodes
        sorted_eligible_nodes = sorted(eligible_nodes.keys(), key=lambda k: eligible_nodes[k], reverse=True)
        for i in range(remainder):
            key_to_add = sorted_eligible_nodes[i % len(sorted_eligible_nodes)]
            allocation[key_to_add] += 1
    elif remainder < 0:
        # If there are too many trials, remove them one by one from the lowest-scored nodes
        sorted_eligible_nodes = sorted(eligible_nodes.keys(), key=lambda k: eligible_nodes[k])
        for i in range(abs(remainder)):
            key_to_remove = sorted_eligible_nodes[i % len(sorted_eligible_nodes)]
            if allocation[key_to_remove] > 0:
                allocation[key_to_remove] -= 1

    return allocation


if __name__=='__main__':
    parser = argparse.ArgumentParser(description = "Trial 策略選擇，有以下選項😍")
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

    try:
        with open(os.path.join(dir_path, 'score.json'),'r') as score_json_file:
            score_json_data_original = json.load(score_json_file)
    except FileNotFoundError:
        print(f"Error: score.json not found at {os.path.join(dir_path, 'score.json')}. Exiting.")
        exit(1)


    trial_allocation = allocate_trials_by_score(score_json_data_original, HYPER_NUM)
    print("[Trial Allocation by Score]", trial_allocation)

    new_json = {"CPU": {}, "GPU": {}}
    for key, num_trials in trial_allocation.items():
        ip, node_type = key.split('_')
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
        out_result.write(f"Resource allocation from score.json: {score_json_data_original} \n")
        out_result.write(f"Calculated trial allocation: {trial_allocation} \n")


    model_types = ["resnet-18"]

    for model in model_types:
        with open(os.path.join(LOG_DIR, "Running_Results.txt"), "a+") as out_result:
            out_result.write(f"model_type: {model} \n")

        avg_run_time = 0
        avg_accuracy = 0

        for i in range(args.exp_times):
            if ray.is_initialized():
                ray.shutdown()
                time.sleep(5) # Give Ray time to shut down cleanly
            ray.init(address="ray://"+HEAD_NODE_IP+":10001", runtime_env=runtime_env)
            print("Ray available resources:", ray.available_resources())

            tt = time.ctime()
            tt_tmp = tt.split()
            json_path = os.path.join(LOG_DIR, "results", f"{tt_tmp[-1]}-{tt_tmp[1]}-{tt_tmp[2]}_{tt_tmp[3].replace(':', '-')}_run{i+1}/")

            os.makedirs(json_path, exist_ok=True)
            print(f'{json_path = }')

            # 建立Tuner，並傳入 log_dir 和共享的 comm_log_filename
            tuner_head = Tuner.remote(
                hyper_num = HYPER_NUM,
                model_type = model,
                resource_allocation = score_json_data,
                stop_acc = STOP_ACC,
                stop_iteration = STOP_ITER,
                checkpoint_interval = INTERVAL_CHECK,
                path = json_path,
                hyperparam_mutations = {
                    "lr": (0.0001, 1),
                    "momentum": (0.0001, 1),
                    "batch_size": BATCH_SIZE
                },
                trialmode = MODE_NAME,
                log_dir = LOG_DIR,
                comm_log_filename = comm_log_filename, # 傳入本次執行共享的唯一檔案名
            )

            tuner_head.set_head.remote(tuner_head)

            reporter_actor = Reporter.remote(
                tuner_head,
                max_report_frequency = INTERVAL_REPORT,
                hyper_num = HYPER_NUM,
            )

            while not ray.get(tuner_head.is_finish.remote()):
                time.sleep(1)

            max_acc_index, max_acc, perturbs = ray.get(tuner_head.get_best_accuracy.remote())
            start_time = ray.get(tuner_head.get_start_time.remote())
            run_duration = time.time() - start_time
            avg_run_time += run_duration
            avg_accuracy += max_acc
            resource = ray.get(tuner_head.get_resource.remote())

            # 取得各節點歷史使用的 batch size 並寫入 Running_Results.txt
            node_bs = ray.get(tuner_head.get_node_batch_sizes_history.remote())
            # ... (writing results to file)

            # Clean up actors
            # ray.kill(reporter_actor) # <-- REMOVE THIS LINE
            # ray.kill(tuner_head)     # <-- REMOVE THIS LINE

            ray.shutdown()
            time.sleep(10)

        with open(os.path.join(LOG_DIR, "Running_Results.txt"), "a+") as out_result:
            out_result.write(f"\n--- Final Average Results for {args.exp_times} runs ---\n")
            out_result.write(f"Avg_total_runtime : {avg_run_time/args.exp_times:.2f} \n")
            out_result.write(f"Avg_accuracy : {avg_accuracy/args.exp_times:.4f} \n\n")
