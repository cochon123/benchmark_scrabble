import subprocess
import tarfile


with tarfile.open("/content/v8_joint_bundle.tar.gz", "r:gz") as archive:
    archive.extractall("/content/v8_joint", filter="data")
subprocess.run(
    ["python", "-c", "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"],
    check=True,
)
