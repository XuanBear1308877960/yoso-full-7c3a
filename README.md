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

This repository provides the trained model, reference measurements, and a subset of test trajectories for validation. The remaining data are withheld because they are being used in ongoing research. The complete dataset will be released after that research concludes.

本仓库提供训练好的模型、参考测量数据以及部分测试轨迹，以便进行验证。其余数据仍用于正在进行的研究，暂不公开；待研究结束后，将开源完整数据集。

`YOSO_Data.zip` contains the reference radio map and six trajectories: three from the main test set and three from the Wi-Fi AP-relocation experiment, selected by the lowest Full median localization error within each condition. Each trajectory includes Wi-Fi, CS, and IMU observations, position ground truth, Full predictions, and initialization metadata. AP and anchor identifiers are anonymous; all timestamps are relative to the segment start. `index.csv` lists the selected trajectories and their errors.
