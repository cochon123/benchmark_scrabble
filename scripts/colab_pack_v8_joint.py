import shutil


archive = shutil.make_archive(
    "/content/v8_joint_results", "gztar", "/content/v8_joint/artifacts/v8", "joint"
)
print(archive)
