# import necessary libraries
import pandas as pd
import time
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def run_apriori(filename):
    df = pd.read_csv(filename).iloc[: , 1:]
    time_1 = time.time()
    frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
    print(frequent_itemsets.sort_values('support', ascending=False))
    time_2 = time.time()
    # rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.9)
    # print(rules)
    time_3 = time.time()
    return time_2 - time_1 #, time_3 - time_2


file_prefix = "g_data_"
n_num = [5, 10, 15, 20, 25]
iterations = 20

def run():
    with open(f"{file_prefix}freq_result.csv", "w") as f:
        f.write("iteration,item_5,item_10,item_15,item_20,item_25\n")
        for i in range(iterations):
            print(f"Iteration {i+1} ================================")
            t = []
            for n in n_num:
                tn = run_apriori(f"{file_prefix}{n}.csv")
                print(f"items: {n} => {tn}")
                t.append(str(tn))
            f.write(f"{i+1},{','.join(t)}\n")

print(run_apriori(f"{file_prefix}{5}.csv"))