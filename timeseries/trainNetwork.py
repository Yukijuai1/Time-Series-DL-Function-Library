import torch
import numpy as np


class MSLEoss(torch.nn.Module):
    def __init__(self):
        super(MSLEoss, self).__init__()

    def forward(self, output, label):
        return torch.mean(torch.pow(torch.log(output) - torch.log(label.float() + 1), 2))


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(
                    f'    EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f'    Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.best_model = model
        self.val_loss_min = val_loss


def trainNetwork(data, label, layers, options):

    train_scale = options.train_scale
    test_scale = options.test_scale
    val_scale = options.val_scale
    batch_size = options.batch_size
    learning_rate = options.learning_rate
    epochs = options.epochs
    loss_function = options.loss_function
    optimizer = options.optimizer
    device = options.device
    shuffle = options.shuffle
    verbose = options.verbose
    verbose_frequency = options.VerboseFrequency
    output_network = options.OutputNetwork
    validation_patience = options.ValidationPatience

    # Split data into train, test and validation sets
    train_size = int(len(data) * train_scale)
    test_size = int(len(data) * test_scale)
    val_size = int(len(data) * val_scale)

    train_data = data[:train_size]
    train_label = label[:train_size]

    test_data = data[-test_size:]
    test_label = label[-test_size:]

    val_data = data[train_size:train_size+val_size]
    val_label = label[train_size:train_size+val_size]

    # Create the network
    model = torch.nn.Sequential()
    for i in range(len(layers)):
        model.add_module('Layer1' + str(i+1), layers[i])

    # Define loss function and optimizer
    if loss_function == 'MSE':
        loss_func = torch.nn.MSELoss()
    elif loss_function == 'RMSE':
        loss_func = RMSELoss()
    elif loss_function == 'L1' or loss_function == 'MAE':
        loss_func = torch.nn.L1Loss()
    elif loss_function == 'CrossEntropy':
        loss_func = torch.nn.CrossEntropyLoss()
    elif loss_function == 'MSLE':
        loss_func = MSLEoss()

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    # Create dataloader
    if shuffle == 'never':
        bool_shuffle = False
    else:
        bool_shuffle = True

    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=bool_shuffle)

    test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=bool_shuffle)

    val_dataset = torch.utils.data.TensorDataset(val_data, val_label)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=bool_shuffle)

    if device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Train the network
    print('Training the network...')
    model.to(device)
    model.train()
    if output_network == 'best-validation-loss':
        early_stopping = EarlyStopping(
            patience=validation_patience, verbose=verbose)
    for epoch in range(epochs):
        if shuffle == 'every-epoch':
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=True)

        Train_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = loss_func(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Train_loss += loss.item()*batch_size
        Train_loss = Train_loss / train_size

        model.eval()
        with torch.no_grad():
            Val_loss = 0
            for i, (x, y) in enumerate(val_loader):
                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)

                val_loss = loss_func(y_pred, y)
                Val_loss += val_loss.item() * batch_size
            Val_loss = Val_loss / val_size
            if output_network == 'best-validation-loss':
                early_stopping(Val_loss, model)
                if early_stopping.early_stop:
                    print("    Early stopping")
                    break
        model.train()
        if verbose and (epoch+1) % verbose_frequency == 0:
            print('    Epoch: {}/{}, Train Loss: {:.4f}'.format(epoch +
                  1, epochs, Train_loss))
            print(
                '    Epoch: {}/{}, Val Loss: {:.4f} \n'.format(epoch+1, epochs, Val_loss))

    # Test the network
    print('Testing the network...')
    model.eval()
    with torch.no_grad():
        Test_loss = 0
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = loss_func(y_pred, y)
            Test_loss += loss.item()*batch_size
        Test_loss = Test_loss / test_size
        if verbose:
            print('    Test Loss: {:.4f}'.format(Test_loss))
    return model.cpu()
