data = []

with open('train_list_1.csv', 'r') as f:
    title = f.readline()
    for line in f.readlines():
        token = line[:-1]
        img, ID, cam, _ = token.split(',')
        #gg = img + ',' + ID + ',' + cam + ',2\n'
        #data.append(gg)
        #gg = img + ',' + ID + ',' + cam + ',4\n'
        #data.append(gg)
        gg = img + ',' + ID + ',' + cam + ',8\n'
        data.append(gg)

with open('train_list_8.csv', 'w') as f:
    f.write(title)
    for line in data:
        f.write(line)


data = []

with open('query_list.csv', 'r') as f:
    title = f.readline()
    for line in f.readlines():
        token = line[:-1]
        img, ID, cam, _ = token.split(',')
        #gg = img + ',' + ID + ',' + cam + ',2\n'
        #data.append(gg)
        #gg = img + ',' + ID + ',' + cam + ',4\n'
        #data.append(gg)
        gg = img + ',' + ID + ',' + cam + ',8\n'
        data.append(gg)

with open('query_list_8.csv', 'w') as f:
    f.write(title)
    for line in data:
        f.write(line)
