# RNN in Pytorch using LSTMs

## Usage
`python main.py`

## Loading a model
To add a model that is saved in some file (`model.pt`) do the following:

* Instantiate a model how you would normally. Ie, `model = LSTM_Mod2(...)`, making sure that the parameters of the model are the same as the one you are trying to load (ie, number of hidden layers)
* Call train on the model: `train_los, val_loss = model.train(...)`, and immediately stop the training. This just initializes some fields.
* Do this: `model.load_state_dict(torch.load('model.pt'))`
