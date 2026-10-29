```bash
python -m pip install -r requirement.txt
python main.py --help
python main.py --assets /path/to/assets --events /path/to/events.jsonl --initial-x 0 --initial-y 0 --output predictions.jsonl
```

```python
import torch
from fullmodel import FullModel

model = FullModel()
model.load_state_dict(torch.load("fullmodel.pt", map_location="cpu", weights_only=True)["state_dict"])
model.eval()
```

This repository provides the trained model, reference measurements, and a subset of test trajectories for validation. The remaining data are being used in ongoing research and are not yet publicly available. The complete dataset will be released once this research is completed.

`YOSO_Data.zip` contains the reference radio map and six trajectories: three from the main test set and three from the Wi-Fi AP-relocation experiment. Each trajectory includes Wi-Fi, CS, and IMU observations, position ground truth, Full predictions, and initialization metadata. AP and anchor identifiers are anonymous; all timestamps are relative to the segment start. `index.csv` lists the selected trajectories and their errors.
