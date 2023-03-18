import csv
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_prefix = "g_data_"

def generate_subset(data, size):
    subsets = []
    for i in range(2**size):
        subset = []
        for j in range(size):
            if i & (1 << j):
                subset.append(data[j])
        if len(subset) > 0:
            subsets.append(" ".join(subset))
    return subsets


def average_median_num_item(n):
    iteration = 0
    count = 0
    count_list = []
    with open(f"{_prefix}{n}.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            iteration += 1
            if iteration == 1:
                continue
            temp_count = 0
            for i in range(1, n+1):
                if row[i] == "True":
                    count += 1
                    temp_count += 1
            count_list.append(temp_count)
    return count / 1000, np.median(count_list)


def time_for_generating_subset(size):
    data = list(map(lambda x: str(x), range(size)))
    t1 = time.time()
    subset = generate_subset(data, size)
    t2 = time.time()
    # print(len(subset))
    return t2 - t1


def statistics():
    nx = [5, 10, 15, 20, 25]
    with open("brute_force_data.txt", "w") as f:
        for n in nx:
            ave_num, med_num = average_median_num_item(n)
            average_info = f"average # items for {str(n).rjust(2)}:    {math.floor(ave_num)}\n"
            median_info = f"median # items for {str(n).rjust(2)}:    {math.floor(med_num)}\n"
            f.write(average_info)
            f.write(median_info)
            print(average_info)
            average_info_time = f"time to generate subset for {str(math.floor(ave_num)).rjust(2)}: {'{:.6f}'.format(time_for_generating_subset(math.floor(ave_num)))} seconds\n"
            median_info_time = f"time to generate subset for {str(math.floor(med_num)).rjust(2)}: {'{:.6f}'.format(time_for_generating_subset(math.floor(med_num)))} seconds\n"
            f.write(average_info_time)
            f.write(median_info_time)
            print(average_info_time)


def run_alg(n):
    items_idx = {}
    itemsets = []
    itemset_count = {}
    # print(itemset_count)
    iteration = 0
    with open(f"{_prefix}{n}.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            iteration += 1
            if iteration == 1:
                for i in range(1, n+1):
                    items_idx[i] = row[i]
                itemsets = generate_subset(list(items_idx.values()), n)
                itemset_count = { k : 0 for k in itemsets}
                continue
            row_item = []
            row_sets = []
            for i in range(1, n+1):
                if row[i] == "True":
                    row_item.append(items_idx[i])
            for i in generate_subset(row_item, len(row_item)):
                itemset_count[i] += 1
    # print(items_idx)
    # sorted_count = sorted(itemset_count.items(), key=lambda item: item[1], reverse=True)
    # for k, v in sorted_count:
        # print(f"{v / 1000}          {k}")


def run_alg2_load_data(filename, size):
    items_idx = {}
    data = []
    iteration = 0
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            iteration += 1
            if iteration == 1:
                for i in range(1, size+1):
                    items_idx[i] = row[i]
                continue
            row_item = []
            for i in range(1, size+1):
                if row[i] == "True":
                    row_item.append(items_idx[i])
            data.append(row_item)
    return data, list(items_idx.values())


def run_alg2(n):
    data, uni = run_alg2_load_data(f"{_prefix}{n}.csv", n)
    t1 = time.time()
    # generate 2^d subsets
    subsets = {k: 0 for k in generate_subset(uni, n)}
    for l in data:
        for l_subset in generate_subset(l, len(l)):
            subsets[l_subset] += 1
    t2 = time.time()
    print(f"alg2 time for {str(n).rjust(2)}:   {t2-t1}")
    # sorted_count = sorted(subsets.items(), key=lambda item: item[1], reverse=True)
    # for k, v in sorted_count:
    #     print(f"{v / 1000}          {k}")


def run_alg3(n):
    data, uni = run_alg2_load_data(f"{_prefix}{n}.csv", n)
    t1 = time.time()
    subsets = {k: 0 for k in generate_subset(uni, n)}
    for key in subsets.keys():
        key_list = key.split(" ")
        key_list_len = len(key_list)
        for l in data:
            key_list_count = 0
            for i in l:
                if i in key_list:
                    key_list_count += 1
            if key_list_count == key_list_len:
                subsets[key] += 1
    t2 = time.time()
    # print(f"alg3 time for {str(n).rjust(2)}:   {t2-t1}")
    # sorted_count = sorted(subsets.items(), key=lambda item: item[1], reverse=True)
    # for k, v in sorted_count:
    #     print(f"{v / 1000}          {k}")
    return t2 - t1


def alg3_stat(n):
    alg3_info = f"{'Complexity:'.ljust(30)}2^n * txn * avg"
    avg, med = average_median_num_item(n)
    t = run_alg3(n)
    alg3_complexity = pow(2, n) * 1000 * math.floor(avg)
    alg3_c = t / alg3_complexity

    average_info = f"{f'average # items for {str(n).rjust(2)}:'.ljust(30)}{math.floor(avg)} items"
    median_info = f"{f'median # items for {str(n).rjust(2)}:'.ljust(30)}{math.floor(med)}"
    time_info = f"{f'alg3 time for {str(n).rjust(2)}:'.ljust(30)}{t} seconds"
    complexity_info = f"{f'alg3 complexity for {str(n).rjust(2)}:'.ljust(30)}{alg3_complexity}"
    c_info = f"{f'alg3 constant for {str(n).rjust(2)}:'.ljust(30)}{alg3_c}"

    print(alg3_info)
    print(average_info)
    print(time_info)
    print(complexity_info)
    print(c_info)

    return math.floor(avg), t, alg3_complexity, alg3_c


def save_alg3():
    nx = [5, 10, 15]
    iteration = 10
    with open("alg3_stat.csv", "w") as f:
        f.write("n_item,average_item,time,complexity,constant_c\n")
        for n in nx:
            print("=====================")
            for i in range(iteration):
                avg, t, complexity, c = alg3_stat(n)
                f.write(f"{n},{avg},{t},{complexity},{c}\n")


def get_brute_force_c():
    df = pd.read_csv("alg3_stat.csv")
    return df["constant_c"].mean()


def get_brute_force_complexity(c, n, avg):
    return pow(2, n) * 1000 * avg * c


def get_apriori_stat():
    df = pd.read_csv("g_data_freq_result.csv")
    df_list = [df[f'item_{n}'] for n in [5, 10, 15, 20, 25]]
    y_mean = np.array(list(map(lambda x: x.mean(), df_list)))
    y_std = np.array(list(map(lambda x: x.std(), df_list)))
    return y_mean, y_std


def plot_graph():
    x = [5, 10, 15, 20, 25]
    avg_dict = {5: 3, 10: 7, 15: 11, 20: 15, 25: 18}
    c = get_brute_force_c()

    y_b = list(map(lambda x: get_brute_force_complexity(c, x, avg_dict[x]), x))
    print(c)
    print(y_b)

    y_a_mean, y_a_std = get_apriori_stat()
    print(y_a_mean)
    print(y_a_std)

    # Plot the first graph
    # plt.plot(x, y_a_mean, label="apriori")
    # plt.fill_between(x, y_a_mean - 2 * y_a_std, y_a_mean + 2 * y_a_std, alpha=0.9, color="red", label="apriori - 2 std")
    # plt.plot(x, y_b, label="brute-force", color="orange")
    # plt.xlabel('Number of Unique Items')
    # plt.ylabel('Time (Seconds)')
    # plt.title('Time Spent in Generating Frequent Itemsets')

    # plt.ylim(-20, 1000)
    # plt.xticks(x)
    # plt.legend()
    # plt.show()
    

# statistics()

# t1 = time.time()
# run_alg(5)
# t2 = time.time()
# print(t2 - t1)

plot_graph()


