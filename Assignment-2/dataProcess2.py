import random
import csv

iteration = 1000
num_items = [5, 10, 15, 20, 25]
# least_pick = [2, 4, 6, 8, 10]
unique_items = ['milk', 'bread', 'egg', 'butter', 'cheese', 'yogurt', 'cereal', 'rice', 'pasta', 'sugar', 'flour', 'salt', 'pepper', 'garlic', 'onion', 'potato', 'carrot', 'lettuce', 'tomato', 'apple', 'banana', 'orange', 'grape', 'lemon', 'lime']

print(len(unique_items))

for idx, n in enumerate(num_items):
    uni_items = unique_items[:n]
    with open(f"g_data_{n}.csv", "w") as f:
        f.write(f"txn,{','.join(uni_items)}\n")
        for i in range(iteration):
            default_list = ["False"] * n
            pick_num = random.randrange(round(n/2), n + 1)
            random_list = random.sample(range(n), pick_num)
            for j in random_list:
                default_list[j] = "True"
            f.write(f"{i+1},{','.join(default_list)}\n")

print("done")
