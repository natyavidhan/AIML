import numpy as np

images = open('assets/t10k-images.idx3-ubyte', 'rb')
labels = open('assets/t10k-labels.idx1-ubyte', 'rb')

images.read(16)
labels.read(8)

data = [["label"] + ["pixel{}".format(i) for i in range(784)]]

for n in range(10000):
    image_data = images.read(28 * 28)
    label_data = labels.read(1)

    image = np.frombuffer(image_data, dtype=np.uint8).reshape(28, 28)
    label = np.frombuffer(label_data, dtype=np.uint8)

    row = [int(label[0])] + [int(image[i, j]) for i in range(28) for j in range(28)]
    data.append(row)
    
with open("mnist.csv", "w") as f:
    for row in data:
        f.write(",".join(map(str, row)) + "\n")