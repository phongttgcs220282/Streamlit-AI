import pandas as pd  # Ensure pandas is properly imported


def get_summary():
    try:
        df = pd.read_csv("dataset.csv")
        return df.describe().to_dict()
    except FileNotFoundError:
        return {"error": "File dataset.csv not found."}
    except pd.errors.EmptyDataError:
        return {"error": "File dataset.csv is empty."}
    except Exception as e:
        return {"error": str(e)}
