import numpy as np
from scipy import stats
from scipy.spatial.distance import mahalanobis
import math


def manhattan(v1, v2):
    l1_norm = np.linalg.norm(v1 - v2, ord=1)
    print(f"manhattan:              {l1_norm}")

def euclidean(v1, v2):
    l2 = np.linalg.norm(v1 - v2)
    print(f"eculidean:              {l2}")

def supremum(v1, v2):
    print(f"supremum:               {np.linalg.norm(v1 - v2, ord=np.inf)}")

def minkowski(v1, v2, p):
    print(f"minkowski:              {np.power(np.sum(np.power(np.abs(v1 - v2), p)), 1/p)}")

def cosine_similarity(v1, v2):
    print(f"cosine similarity:      {np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))}")

def consine_distance(v1, v2):
    print(f"consine distance:       {1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))}")

def simple_matching_coefficient_SMC(v1, v2):
    a = np.sum((v1 == v2) & (v1 == 1))
    b = np.sum((v1 != v2) & (v1 == 0))
    c = np.sum((v1 != v2) & (v1 == 1))
    d = np.sum((v1 == v2) & (v1 == 0))
    smc = (a + d) / (a + b + c + d)
    print(f"SMC:                    {smc}")

def jaccard_coefficient(v1, v2):
    a = np.sum((v1 == v2) & (v1 == 1))
    b = np.sum((v1 != v2) & (v1 == 0))
    c = np.sum((v1 != v2) & (v1 == 1))
    jaccard = a / (a + b + c)
    print(f"jaccard:                {jaccard}")

def hamming(v1, v2):
    print(f"hamming:                {np.sum(v1 != v2)}")

def median_mode_mean_variance_range_IR(v):
    print(f"median:                 {np.median(v)}")
    print(f"mode:                   {stats.mode(v).mode[0]}")
    print(f"mean:                   {np.mean(v)}")
    print(f"variance:               {np.var(v)}")
    print(f"range:                  {np.max(v) - np.min(v)}")
    q1, q3 = np.percentile(v, [25, 75])
    print(f"interquartile range:    {q3 - q1}")


def mahalanobis_distance(v1, v2, cov):
    print(f"mahalanobis:            {mahalanobis(v1, v2, np.linalg.inv(cov))}")

def get_gini(classes, total):
    return 1 - np.sum(list(map(lambda x: (x / total)**2, classes))).item()

def gini(classes, total):
    print(f"gini:                   {get_gini(classes, total)}")

def gini_split(classes):
    ttl = np.sum(list(map(np.sum, classes))).item()
    ginisplit = np.sum(list(map(lambda x: get_gini(x, np.sum(x).item()) * (np.sum(x).item() / ttl), classes)))
    print(f"gini split:             {ginisplit}")

def get_entropy(classes):
    ttl = np.sum(classes).item()
    return 0 - np.sum(list(map(lambda x: (x / ttl) * math.log2(x / ttl) if x != 0 else 0, classes)))

def entropy(classes):
    print(f"entropy:                {get_entropy(classes)}")

def error(classes):
    ttl = np.sum(classes).item()
    val = 1 - np.max(list(map(lambda x: x / ttl, classes))).item()
    print(f"error:                  {val}")

def entropy_gain(classes):
    vecs = np.sum(classes, axis=0)
    ttl = np.sum(vecs).item()
    original = get_entropy(vecs)
    print(f"entropy gain:           {original - np.sum(list(map(lambda x: get_entropy(x) * (np.sum(x).item() / ttl), classes))).item()}")

def _95_confidence(model1, model2):
    n1 = model1[0]
    e1 = model1[1]
    n2 = model2[0]
    e2 = model2[1]
    d = abs(e1 - e2)
    val = e1 * (1 - e1) / n1 + e2 * (1 - e2) / n2
    val1 = d - 1.96 * math.sqrt(val)
    val2 = d + 1.96 * math.sqrt(val)
    print(f"95% interval:           ({val1}, {val2})")

def rule_eveluation(n, covered_n, k_classes, probability):
    print(f"accuracy:               {covered_n / n}")
    print(f"laplace:                {(covered_n + 1) / (n + k_classes)}")
    print(f"m-estimate:             {(covered_n + k_classes * probability) / (n + k_classes)}")

def normal_distribution_probability(values, target):
    mean = np.mean(values).item()
    variance = np.var(values).item()
    print(f"normal distribute probability:  {1 / (math.sqrt(2 * math.pi * variance)) * math.exp(0 - (target - mean)**2 / (2 * variance))}")

def support_confidence(s1, s2, ttl):
    print(f"support:                {s2 / ttl}")
    print(f"confidencd:             {s2 / s1}")


# ***************    EXAMPLES    *************** #

vec1 = np.array([ 3, 2, 0, 2, 1, 1, 0, 2, 1, 0 ])
vec2 = np.array([ 1, 2, 0, 1, 1, 0, 2, 1, 3, 1 ])
print(f"v1: {vec1}\nv2: {vec2}")

manhattan(vec1, vec2)
euclidean(vec1, vec2)
supremum(vec1, vec2)
cosine_similarity(vec1, vec2)
minkowski(vec1, vec2, 2)

# ********************************************** #
print("=" * 50)

vec3 = np.array([1,0,1,1,1,0,1])
vec4 = np.array([1,0,0,1,0,0,1])
print(f"v3: {vec3}\nv4: {vec4}")

simple_matching_coefficient_SMC(vec3, vec4)
jaccard_coefficient(vec3, vec4)
hamming(vec3, vec4)

# ********************************************** #
print("=" * 50)

vec5 = np.array([3, 7, 9, 2, 1, 5, 6, 3])
print(f"v5: {vec5}")

median_mode_mean_variance_range_IR(vec5)

# ********************************************** #
print("=" * 50)

vec6 = np.array([0.5, 0.5])
vec7 = np.array([1.5, 1.5])

covariance = np.array([[0.3, 0.2],[0.2, 0.3]])

print(f"v6: {vec6}\nv7: {vec7}")
mahalanobis_distance(vec6, vec7, covariance)

# ********************************************** #
# find least gini split
print("=" * 50)

vec8 = np.array([4, 2])
vec9 = np.array([1, 3])
print(f"v8: {vec8}\nv9: {vec9}")

gini8 = gini(vec8, np.sum(vec8).item())
gini9 = gini(vec9, np.sum(vec9).item())
gini_split(np.array([vec8, vec9]))

vec10 = np.array([1,4])
vec11 = np.array([2,1])
vec12 = np.array([1,1])
print(f"v10: {vec10}\nv11: {vec11}\nv12: {vec12}")

gini_split(np.array([vec10, vec11, vec12]))

# ********************************************** #
print("=" * 50)

vec13 = np.array([0,6])
vec14 = np.array([1,5])
vec15 = np.array([2,4])
print(f"v13: {vec13}\nv14: {vec14}\nv15: {vec15}")
entropy(vec13)
error(vec13)
entropy(vec14)
error(vec14)
entropy(vec15)
error(vec15)
entropy_gain([vec13, vec14, vec15])

# ********************************************** #
print("=" * 50)

# [size, error]
_95_confidence([30, 0.15], [5000, 0,25])

# ********************************************** #
print("=" * 50)

vec16 = np.array([125, 100, 70, 120, 60, 220, 75])
targe16 = 120
normal_distribution_probability(vec16, targe16)

# ********************************************** #
# Mining Association Rules
print("=" * 50)

# given {A, B} -> {C}
# support 1 count: {A, B}     given 3
# support 2 count: {A, B, C}  given 3
# ttl:                  given 5
support_confidence(3, 3, 5)

