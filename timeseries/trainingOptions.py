
class TrainingOptions:
    def __init__(self, train_scale, test_scale, val_scale, batch_size, learning_rate, epochs, loss_function, optimizer, device, shuffle, verbose,VerboseFrequency,OutputNetwork,ValidationPatience):
        self.train_scale = train_scale
        self.test_scale = test_scale
        self.val_scale = val_scale
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.shuffle = shuffle
        self.verbose = verbose
        self.VerboseFrequency = VerboseFrequency
        self.OutputNetwork = OutputNetwork
        self.ValidationPatience = ValidationPatience

    def print_var(self):
        print("--train_scale",self.train_scale,"--test_scale",self.test_scale,"--val_scale",self.val_scale,"--batch_size",self.batch_size,"--learning_rate",self.learning_rate,"--epochs",self.epochs,"--loss_function",self.loss_function,"--optimizer",self.optimizer,"--device",self.device,"--shuffle",self.shuffle,"--verbose",self.verbose,"--VerboseFrequency",self.VerboseFrequency,"--OutputNetwork",self.OutputNetwork,"--ValidationPatience",self.ValidationPatience)


def trainingOptions(train_scale, test_scale, val_scale, batch_size=32, learning_rate=0.001, epochs=30, loss_function='MSE', optimizer='adam', device='cpu', shuffle='never', verbose=0,VerboseFrequency=5,OutputNetwork='last-iteration',ValidationPatience=5):
    return TrainingOptions(train_scale, test_scale, val_scale, batch_size, learning_rate, epochs, loss_function, optimizer, device, shuffle, verbose,VerboseFrequency,OutputNetwork,ValidationPatience)
