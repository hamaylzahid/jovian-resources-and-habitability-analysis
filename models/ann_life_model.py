import numpy as np

def estimate_life_longevity(row):

    if row['Probability of Life Presence (%)'] < 10:
        return 0

    elif row['Radiation levels (mSv/year)'] > 0.1:
        return np.random.randint(1,50)

    elif row['Temperature range (C)'] < -150:
        return np.random.randint(10,500)

    elif row['Water presence (Y/N)'] == 1:
        return np.random.randint(100,10000)

    else:
        return np.random.randint(50,1000)
