import pandas as pd
from pathlib import Path

class Debugger:
    def __init__(self, debug_path) -> None:
        self.debug_path = debug_path
        Path(self.debug_path).mkdir(parents=True, exist_ok=True)
        
    def debug_series(self, name, *series):
        pd.DataFrame(zip(*series)).to_csv(self.debug_path + "/" + name + ".csv", index=False)

    def debug_str(self, s):
        with open(self.debug_path + "/" + "log.txt", "a") as f:
            f.write(str(s) + "\n")


class FakeDebugger:
    def __init__(self) -> None:
        pass
    
    def debug_series(self, name, series1, series2):
        pass