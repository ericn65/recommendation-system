import pandas as pd


def clean_wrong_numerical_types(input_data: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Checks the data columns where numerical data is corrupted.
    Analyses if a `.` should be change `,` and similar.

    Parameters
    ----------
    input_data : pd.DataFrame
        The data to be filtered.
    rules : dict
        The rules to be checked using a config file as a dict.

    Returns
    -------
    output_data : pd.DataFrame
        The data uniformed for numerical aspects using the rules config.
    """
    output_data: pd.Dataframe = ""

    return output_data


def check_duplicated(input_data: pd.DataFrame, data_id: str) -> pd.DataFrame:
    """Checks duplicated stuff."""
