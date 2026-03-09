class Callback:

    def on_train_start(self, trainer): pass

    def on_train_end(self, trainer): pass


    def on_epoch_start(self, trainer): pass

    def on_epoch_end(self, trainer): pass


    def on_batch_start(self, trainer): pass

    def on_batch_end(self, trainer): pass


    def on_forward_end(self, trainer): pass

    def on_backward_end(self, trainer): pass