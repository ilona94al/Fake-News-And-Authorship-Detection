from keras.callbacks import Callback


class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """

    def __init__(self, progress_bar, epochs):
        Callback.__init__(self)
        self.progress_bar = progress_bar
        self.epochs = epochs

    def on_train_begin(self, logs={}):
        self.progress_bar.setValue(1)
        self.progress_bar.update()

    def on_epoch_begin(self, epoch, logs={}):
        progress_percent = int(100 * epoch / self.epochs) + 1
        self.progress_bar.setValue(progress_percent)
        self.progress_bar.update()

    # def on_epoch_end(self, epoch, logs={}):
    #     progress_percent = int(100 * (epoch + 1) / self.epochs)
    #     self.progress_bar.setValue(progress_percent)
    #     self.progress_bar.update()

