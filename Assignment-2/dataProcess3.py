import csv
import random

def get_value(tx, ls):
    res = ""
    for i in ls:
        res += ",True" if i in tx else ",False"
    return res

txn = {}
items = {}
i = 0

with open("Groceries_dataset.csv", "r") as f:
    reader = csv.reader(f)
    for line in reader:
        i += 1
        if i == 1:
            continue
        t = line[0]
        item = line[2].strip()
        if t not in txn:
            txn[t] = []
        txn[t].append(item)

        if item not in items.keys():
            items[item] = 1
        else:
            items[item] += 1
    print("rules loaded...")

items_list = {k: v for k, v in sorted(items.items(), key=lambda item: item[1], reverse=True)[:25]}
keys = [k for k, v in sorted(items_list.items(), key=lambda item: item[1], reverse=True)]


# item_num = [5, 10, 15, 20]
item_5 = keys[:5]
item_10 = keys[:10]
item_15 = keys[:15]
item_20 = keys[:20]
item_25 = keys[:25]

i_5 = 0
i_10 = 0
i_15 = 0
i_20 = 0
i_25 = 0

threshold = 1000

with open("gd_data_5.csv", "w") as f1, \
    open("gd_data_10.csv", "w") as f2, \
    open("gd_data_15.csv", "w") as f3, \
    open("gd_data_20.csv", "w") as f4, \
    open("gd_data_25.csv", "w") as f5:
    f1.write(f"txn,{','.join(item_5)}\n")
    f2.write(f"txn,{','.join(item_10)}\n")
    f3.write(f"txn,{','.join(item_15)}\n")
    f4.write(f"txn,{','.join(item_20)}\n")
    f5.write(f"txn,{','.join(item_25)}\n")

    for key in txn.keys():
        v_5 = get_value(txn[key], item_5)
        if "True" in v_5 and i_5 < threshold:
            i_5 += 1
            f1.write(f"{i_5}{v_5}\n")
        
        v_10 = get_value(txn[key], item_10)
        if "True" in v_10 and i_10 < threshold:
            i_10 += 1
            f2.write(f"{i_10}{v_10}\n")

        v_15 = get_value(txn[key], item_15)
        if "True" in v_15 and i_15 < threshold:
            i_15 += 1
            f3.write(f"{i_15}{v_15}\n")

        v_20 = get_value(txn[key], item_20)
        if "True" in v_20 and i_20 < threshold:
            i_20 += 1
            f4.write(f"{i_20}{v_20}\n")
        
        v_25 = get_value(txn[key], item_25)
        if "True" in v_25 and i_25 < threshold:
            i_25 += 1
            f5.write(f"{i_25}{v_25}\n")
