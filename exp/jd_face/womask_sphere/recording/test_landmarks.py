from pyhocon import ConfigFactory
from models.dataset import Dataset

conf = ConfigFactory.parse_file("./confs/womask.conf")

# Grab the nested dataset config (this is what Dataset() actually receives)
ds_conf = conf["dataset"]

# Force-replace CASE_NAME inside the nested key that Dataset reads
old = str(ds_conf["data_dir"])
ds_conf["data_dir"] = old.replace("CASE_NAME", "jd_face")

# Print BOTH forms so we know what's real
print("Using data_dir (nested):", ds_conf["data_dir"])
try:
    print("Using data_dir (dotted):", conf["dataset.data_dir"])
except Exception as e:
    print("No dotted key dataset.data_dir (ok):", e)

# Initialize dataset with nested config
d = Dataset(ds_conf)

px, names = d.get_landmark_pixels(0)
print("Landmark pixels:", px)
print("Names:", names)
print("Shape:", px.shape if px is not None else None)

ro, rd, _, _ = d.gen_random_landmark_rays_at(0, 10)
print("Rays origin shape:", ro.shape if ro is not None else None)
print("Rays direction shape:", rd.shape if rd is not None else None)
