import argparse
import os
import shutil
import sys
import time
import typing as t

from pathlib import Path

from smartsim import Experiment
from smartsim.entity import Application
from smartsim.settings import LaunchSettings
from smartsim.status import TERMINAL_STATUSES
from smartsim.launchable.job import Job
from smartsim.entity.inference_service import TorchInferenceService

parser = argparse.ArgumentParser("Mock application")
parser.add_argument("--log_max_batchsize", default=8, type=int)
parser.add_argument("--num_nodes_app", default=1, type=int)
parser.add_argument(
    "--toolkit", default="torch", choices=["torch", "tensorflow", "onnx"], type=str
)
parser.add_argument("--wm_node", type=str, default=None)
parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="gpu")
args = parser.parse_args()

NUM_RANKS_PER_NODE = 1
NUM_NODES_APP = args.num_nodes_app
NUM_WORKERS = 1
BATCH_SIZE = 2
BATCH_TIMEOUT = 0.0
filedir = os.path.dirname(__file__)
worker_manager_script_name = os.path.join(filedir, "standalone_worker_manager.py")
if args.toolkit == "torch":
    # keeping old name for backward compatibility
    app_script_name = os.path.join(filedir, "mock_app.py")
else:
    app_script_name = os.path.join(filedir, f"mock_app_{args.toolkit}.py")

transport: t.Literal["hsta", "tcp"] = "hsta"

os.environ["SMARTSIM_DRAGON_TRANSPORT"] = transport

exp_path = os.path.join(
    filedir,
    "benchmark",
    args.toolkit,
    f"throughput_n{NUM_NODES_APP}_rpn{NUM_RANKS_PER_NODE}_timeout{BATCH_TIMEOUT}",
    f"samples{2**args.log_max_batchsize}",
)
try:
    shutil.rmtree(exp_path)
    time.sleep(2)
except Exception:
    pass
os.makedirs(exp_path, exist_ok=True)
exp = Experiment("MLI_benchmark", exp_path=exp_path)

inference_service_ls: LaunchSettings = LaunchSettings("dragon")

aff = []

inference_service_ls.launch_args.set_cpu_affinity(aff)
inference_service_ls.launch_args.set_gpu_affinity([0, 1, 2, 3])
if args.wm_node:
    inference_service_ls.launch_args.set_hostlist([args.wm_node])

inference_service = TorchInferenceService(
    "torch_server",
    launch_settings=inference_service_ls,
    device=args.device,
    num_workers=NUM_WORKERS,
    batch_size=BATCH_SIZE,
    batch_timeout=BATCH_TIMEOUT,
)

inference_job = inference_service.build_jobs()

app_ls: LaunchSettings = LaunchSettings("dragon")
app_ls.launch_args.set_tasks_per_node(NUM_RANKS_PER_NODE)
app_ls.launch_args.set_nodes(NUM_NODES_APP)


app = Application(
    "app",
    sys.executable,
    exe_args=[
        app_script_name,
        "--device",
        args.device,
        "--log_max_batchsize",
        str(args.log_max_batchsize),
    ],
)

app.files.add_copy(Path(app_script_name))

if args.toolkit == "torch":
    model_name = os.path.join(filedir, f"resnet50.{args.device}.pt")
    app.files.add_symlink(Path(model_name), Path(f"resnet50.{args.device}.pt"))

app_job = Job(app, app_ls)

inference_id = exp.start(inference_job)

app_job_id = exp.start(app_job)

while True:
    if exp.get_status(*app_job_id)[0] in TERMINAL_STATUSES:
        time.sleep(10)
        exp.stop(inference_id)
        break
    if exp.get_status(*inference_id)[0] in TERMINAL_STATUSES:
        time.sleep(10)
        exp.stop(app_job_id)
        break
    time.sleep(10)

print("Exiting.")
