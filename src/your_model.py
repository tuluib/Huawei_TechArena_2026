import pathlib
import pandas as pd
import numpy as np

# When you have a real ONNX model, you will uncomment this:
# import onnxruntime as ort

class YourModel:
    def load_model(self, onnx_file: pathlib.Path) -> None:
        """
        Loads the trained model weights. 
        For now, this is a dummy that does nothing.
        """
        # self.session = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
        pass

    def run(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        This is the core processor. 
        INPUT: A pandas DataFrame with columns: BVP, EDA, HR, IBI_dt, IBI_duration, Temperature
        OUTPUT: A pandas DataFrame with a single column 'Activity' (Stress, Activity, Rest)
        """
        
        # 1. The Signal Processing Engineer's filtering code will go here.
        # filtered_data = apply_filters(input_df)
        
        # 2. The ML Architect's model inference will go here.
        # features = extract_features(filtered_data)
        # labels = self.session.run(None, {"input": features})[0]
        
        # 3. For now, this is a DUMMY output that just guesses "Rest" for every row.
        dummy_labels = ["Rest"] * len(input_df)
        
        return pd.DataFrame({"Activity": dummy_labels}, index=input_df.index)

def get_model():
    """Huawei's server will call this exact function to get your chip."""
    model = YourModel()
    
    # Eventually, it will load your real .onnx file here:
    # model.load_model(pathlib.Path(__file__).with_suffix(".onnx"))
    
    return model