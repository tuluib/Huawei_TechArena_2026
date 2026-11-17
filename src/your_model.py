import pathlib
import numpy as np
import pandas as pd
import onnxruntime as ort
from adm_base_class import ADM_Base

REQUIRED_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
BLOCK_SIZE = 640  # 10-second window at 64 Hz
CLASSES = np.array(["Rest", "Activity", "Stress"], dtype=object)


class YourModel(ADM_Base):
    def load_model(self, onnx_file: pathlib.Path) -> None:
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            onnx_file,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def run(self, input: pd.DataFrame) -> pd.DataFrame:
        n_rows = len(input)
        if n_rows == 0:
            return pd.DataFrame({"Activity": pd.Series(dtype=str)}, index=input.index)

        # 1. Fast vectorized preprocessing
        arr = input[REQUIRED_COLS].ffill().fillna(0.0).to_numpy(dtype=np.float32)
        arr = np.clip(arr, -1e4, 1e4)

        # 2. Pad and reshape into 640-row (10-second) blocks
        n_cols = arr.shape[1]
        n_blocks = max(1, int(np.ceil(n_rows / BLOCK_SIZE)))
        pad_len = (n_blocks * BLOCK_SIZE) - n_rows
        if pad_len > 0:
            arr = np.pad(arr, ((0, pad_len), (0, 0)), mode="edge")

        blocks = arr.reshape(n_blocks, BLOCK_SIZE, n_cols)
        features = np.hstack([blocks.mean(axis=1), blocks.std(axis=1)]).astype(np.float32)

        # 3. Run ONNX inference -> returns integer indices (0, 1, 2)
        pred_ids = self.session.run(None, {self.input_name: features})[0]
        block_labels = CLASSES[np.clip(pred_ids, 0, 2)]

        # 4. Broadcast block predictions back to original 64 Hz timestamps
        row_preds = np.repeat(block_labels, BLOCK_SIZE)[:n_rows]
        return pd.DataFrame({"Activity": row_preds}, index=input.index)


def get_model() -> ADM_Base:
    model = YourModel()
    model.load_model(pathlib.Path(__file__).with_suffix(".onnx"))
    return model