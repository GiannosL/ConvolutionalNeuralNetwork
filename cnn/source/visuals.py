import matplotlib.pyplot as plt

def plot_image(input_image):
    plt.imshow(input_image)
    plt.gray()
    plt.show()