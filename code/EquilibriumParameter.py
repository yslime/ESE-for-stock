# Here, we can calculate equilibrium parameter by
# using attribution parameter
import numpy as np
from sklearn.linear_model import LinearRegression


def equilibrium_parameter(part_coeffs, part, whole):
    part_col_total = [0 for _ in range(part.shape[0])]
    for rows in range(part.shape[0]):
        # Save the current sum of each columns
        part_col_total = [a + x_sum_part for a, x_sum_part in zip(a, part_col_total)]

    ep = [0 for _ in range(len(part_coeffs))]
    for i in range(len(part_coeffs)):
        ep[i] = (part_coeffs[i] * part_col_total[i]) / whole

    return ep


## for covid
def equilibrium_state_parameter_set_distribution(attributes, correlations):
    sumDistribution = 0
    for i in range(attributes.shape[1]):
        at = feature_distribution(attributes[i, :], correlations[i])
        sumDistribution += at

    espsDistribution = ((sumDistribution / attributes.shape[1]) + 1) / attributes.shape[0]

    return espsDistribution

####
def feature_distribution(attribute, correlation):
    maxAttr = np.max(attribute)
    minAttr = np.min(attribute)

    diff = attribute - np.mean(attribute)
    featureDistribution = (diff / (maxAttr - minAttr)) * correlation

    return featureDistribution
