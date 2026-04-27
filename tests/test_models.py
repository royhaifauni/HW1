import torch
from src.models.rnn import RNNFilter
from src.models.lstm import LSTMFilter
from src.sdk.gatekeeper import APIGatekeeper


def test_rnn_shapes():
    model = RNNFilter(input_dim=5, hidden_dim=32)
    x = torch.randn(8, 10, 5)  # (batch, seq, features)
    hidden = model.init_hidden(8)
    out, next_hidden = model(x, hidden)

    assert out.shape == (8, 10, 1)
    assert next_hidden.shape == (1, 8, 32)


def test_lstm_shapes():
    model = LSTMFilter(input_dim=5, hidden_dim=32)
    x = torch.randn(8, 10, 5)
    hidden = model.init_hidden(8)
    out, next_hidden = model(x, hidden)

    assert out.shape == (8, 10, 1)
    assert isinstance(next_hidden, tuple)
    assert next_hidden[0].shape == (1, 8, 32)


def test_gatekeeper_inference():
    model = RNNFilter()
    gk = APIGatekeeper(model)
    x = torch.randn(4, 1, 5)
    out = gk.run_inference(x)

    assert out.shape == (4, 1, 1)
    assert not out.requires_grad
