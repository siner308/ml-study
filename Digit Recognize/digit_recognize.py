import pandas as pd

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_answer = train_df['label']
train_df = train_df.drop('label', axis=1)

arr = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for digit in train_answer:
    arr[digit.values - 1] = arr[digit.values - 1] + 1

import numpy as np
import matplotlib.pyplot as plt


def display_digits(N):
    images = np.random.randint(low=0, high=35300, size=N).tolist()

    subset_images = train_df.iloc[images, :]
    subset_images.index = range(1, N + 1)

    for i, row in subset_images.iterrows():
        plt.subplot((N // 8) + 1, 8, i)
        pixels = row.values.reshape((28, 28))
        plt.imshow(pixels, cmap='gray')
        plt.xticks([])
        plt.yticks([])

    plt.show()
    return ""

display_digits(40)

arr = np.random.randint(low=0, high=35300, size=1).tolist()[0]

pixels = train_df.iloc[arr, 0:]
image = train_df.iloc[arr, 0:].values.reshape((28, 28))

pmin, pmax = image.min(), image.max()

print(pmin)
print(pmax)

rescaled_image = 255 * (image - pmin) / (pmax - pmin)
rescaled_pixels = rescaled_image.flatten()

bw_pixels = pixels.apply(lambda x: 0 if x == 0 else 255)
bw_image = bw_pixels.values.reshape((28, 28))

plt.subplot(1, 3, 3)
plt.imshow(bw_image, cmap='gray')
plt.title('black&wite only image')
plt.show()


def display_rescaled_digits(N):
    images = np.random.randint(low=0, high=35300, size=N).tolist()

    subset_images = train_df.iloc[images, :]
    subset_images.index = range(1, N + 1)

    for i, row in subset_images.iterrows():
        plt.subplot((N // 8) + 1, 8, i)
        image = row.values.reshape((28, 28))
        pixels = image.flatten()

        bw_pixels = row.apply(lambda x: 0 if x == 0 else 255)
        bw_image = bw_pixels.values.reshape((28, 28))

        plt.imshow(bw_image, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()

    return ""

display_rescaled_digits(40)