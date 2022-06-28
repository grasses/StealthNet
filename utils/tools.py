

class Utils:
    _instance = None

    def __init__(self):
        self.cache = {}

    @staticmethod
    def _get_instance():
        if Utils._instance is None:
            Utils._instance = Utils()
        return Utils._instance

    @staticmethod
    def show_images(images, labels, title='examples'):
        plt.figure(figsize=(10, 10))
        plt.subplots_adjust(hspace=0.2)
        for n in range(25):
            plt.subplot(5, 5, n + 1)
            img = images[n]
            img = img.numpy().squeeze()
            plt.imshow(img)
            plt.title(f'{labels[n]}')
            plt.axis('off')
        _ = plt.suptitle(title)
        plt.show()

    @staticmethod
    def copy_weights(source_model, target_model):
        # print(source_model.summary())
        # print(target_model.summary())
        for i, layer in enumerate(target_model.layers):
            if not layer.get_weights():
                continue
            source_layer = source_model.get_layer(layer.name)
            # print(layer)
            # print(source_layer)
            layer.set_weights(source_layer.get_weights())
        return target_model

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm