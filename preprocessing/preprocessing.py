import numpy as np

def select_features_for_resource_availability(data):

    selected_features = [
        'Extraction Rate (tons/year)',
        'Replenishment Rate (tons/year)',
        'Crust Stability Index',
        'Operational Risk Factor'
    ]

    return data.dropna(subset=selected_features)


def convert_temperature_range(temp_range):

    if 'to' in str(temp_range):

        lower, upper = map(int, temp_range.split(' to '))

        return (lower + upper) / 2

    return np.nan
